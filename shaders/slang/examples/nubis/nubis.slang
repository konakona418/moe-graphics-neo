// [moe("vertex", "fragment")]

// Nubis3 raymarch: sphere-traces the procedural NVDF (SDF channel), erodes the
// profile with the detail-noise volume, lights it through the precomputed light
// grid, and composites over a Preetham sky. Single full-res pass.

#include "examples/nubis/nubis_common.slang"

struct PushConstants {
    float3 mCameraPos;
    float3 mForward;
    float3 mRight;
    float3 mUp;
    float3 mSunDir;
    float mTanHalfFov;
    float mAspect;
    float mTime;
    float3 mBoxMin;
    float3 mBoxMax;
    float mFarClip;
    float mTransmittanceLimit;
    float mTilingFreq;
    float mDensityScale;
    float mTurbidity;
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] Texture3D<float4> nvdf;
[[vk::binding(1, 0)]] SamplerState linearSampler;
[[vk::binding(2, 0)]] Texture3D<float4> detailNoise;
[[vk::binding(3, 0)]] Texture3D<float4> lightGrid;

struct VertexOutput {
    float4 mPosition : SV_Position;
    float2 mUv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VulkanVertexId) {
    VertexOutput output;
    float2 pos = float2(float((vertexId << 1) & 2), float(vertexId & 2));
    output.mPosition = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    output.mUv = pos;
    return output;
}

struct ModelingData {
    float mProfile;
    float mType;
    float mDensityScale;
    float mSdf;
};

float3 ToSampleCoord(float3 world) {
    return (world - pcs.mBoxMin) / (pcs.mBoxMax - pcs.mBoxMin);
}

bool InBox(float3 uv) {
    return all(uv >= 0.0) && all(uv <= 1.0);
}

ModelingData SampleModeling(float3 uv) {
    float4 n = nvdf.SampleLevel(linearSampler, uv, 0.0);
    ModelingData m;
    m.mProfile = n.x;
    m.mType = n.y;
    m.mDensityScale = n.z;
    m.mSdf = n.w;
    return m;
}

// Nubis detail erosion: blend wispy/billow noise by the detail type, then
// erode the profile and modulate by the density scale.
float DetailDensity(float distance, float3 uv, ModelingData m, bool highFreq) {
    float4 n = detailNoise.SampleLevel(linearSampler, uv * pcs.mTilingFreq, 0.0);

    float wispy = lerp(n.r, n.g, m.mProfile);
    float billowGradient = pow(saturate(m.mProfile), 0.25);
    float billow = lerp(n.b * 0.3, n.a * 0.3, billowGradient);
    float composite = lerp(wispy, billow, m.mType);

    if (highFreq) {
        float hhfWisps = 1.0 - pow(abs(abs(n.g * 2.0 - 1.0) * 2.0 - 1.0), 4.0);
        float hhfBillows = pow(abs(abs(n.a * 2.0 - 1.0) * 2.0 - 1.0), 2.0);
        float hhf = saturate(lerp(hhfWisps, hhfBillows, m.mType));
        float blender = Remap(distance, 50.0, 150.0, 0.9, 1.0);
        composite = lerp(hhf, composite, blender);
    }

    float d = ValueErosion(m.mProfile, composite);
    float powered = pow(saturate(m.mDensityScale), 4.0);
    d *= powered;
    d = pow(d, lerp(0.3, 0.6, max(0.1, powered)));

    if (highFreq) {
        float blender = saturate(Remap(distance, 50.0, 150.0, 0.0, 1.0));
        d = pow(d, lerp(0.5, 1.0, blender)) * lerp(0.666, 1.0, blender);
    }
    return d;
}

float GetDensityToSun(float3 uv, float3 sunDir, float stepLen) {
    float3 extent = pcs.mBoxMax - pcs.mBoxMin;
    float3 duv = sunDir / extent;
    float total = 0.0;
    float3 p = uv + duv * stepLen;
    for (int i = 0; i < 2; ++i) {
        if (InBox(p)) {
            float4 n = nvdf.SampleLevel(linearSampler, p, 0.0);
            total += n.x * n.z;
        }
        p += duv * stepLen;
    }
    if (InBox(p)) {
        total += lightGrid.SampleLevel(linearSampler, p, 0.0).r;
    }
    return total;
}

// Nubis light energy. The light grid gives the optical depth from the sample
// to the sun, so Beer-Lambert transmittance exp(-densityToSun) is the correct
// direct-light term: the sun-facing side is bright, the shadowed side dark.
// (The reference folds this into exp(-attenuation*inScatter*phase*5), which
// with our data reads the shadow side as brighter; this keeps the Nubis phase
// and warm/cool tints but fixes the direction.)
float3 LightEnergy(float cosAngle, float cloudDistance, float densityToSun,
        float densityLod, float sampleProfile, float3 skyNoSun) {
    float sunT = exp(-densityToSun);

    // Henyey-Greenstein phase: forward silver lining plus a broad back lobe
    float phase = max(HgPhase(cosAngle, 0.75), 0.6 * HgPhase(cosAngle, -0.3));
    float direct = saturate(sunT * (0.25 + phase * 1.6));

    // warm sunlit top, cool sky-lit shadow (deliberately darker than the sky)
    float3 sunTint = float3(1.0, 0.96, 0.90);
    float3 shadowTint = lerp(skyNoSun, float3(0.22, 0.34, 0.46),
            saturate(pcs.mSunDir.y + 0.5)) * 0.55;
    float3 cloudColor = lerp(shadowTint, sunTint, direct);

    // multiple scattering lifts the deep interior off pure black
    cloudColor += skyNoSun * (1.0 - sunT) * 0.12;
    return cloudColor;
}

// ---- Preetham sky -------------------------------------------------------

float RayleighPhase(float cosTheta) {
    return (3.0 / (16.0 * MOE_PI)) * (1.0 + pow(cosTheta, 2.0));
}

float SunIntensity(float zenithAngleCos) {
    float cutoffAngle = MOE_PI / 1.95;
    return 1024.0 * max(0.0,
            1.0 - exp(-((cutoffAngle - acos(saturate(zenithAngleCos))) / 6.4)));
}

float3 TotalRayleigh(float3 lambda) {
    return (8.0 * pow(MOE_PI, 3.0) * pow(pow(1.000128, 2.0) - 1.0, 2.0)
            * (6.0 + 3.0 * 0.137))
            / (3.0 * 2.542e25 * pow(lambda, float3(4.0)) * (6.0 - 7.0 * 0.137));
}

float3 TotalMie(float3 lambda, float3 k, float t) {
    float c = 0.2 * t * 10e-18;
    return 0.434 * c * MOE_PI * pow((2.0 * MOE_PI) / lambda, float3(4.007 - 2.0)) * k;
}

float3 SkyColor(float3 rayDir, float3 sunPosition, float turbidity, out float3 noSun,
        out float sunDisk, out float night) {
    float3 up = float3(0.0, 1.0, 0.0);
    // y-up: Frostnova's -sunPosition.z (z-up, negative when the sun is up) maps
    // to +sunPosition.y here.
    float sunfade = 1.0 - saturate(1.0 - exp(sunPosition.y / 450000.0));
    float rayleighCoefficient = 5.75 - (1.0 - sunfade);
    float3 primaries = float3(6.8e-7, 5.5e-7, 4.5e-7);
    float3 betaR = TotalRayleigh(primaries) * rayleighCoefficient;
    float3 betaM = TotalMie(primaries, float3(0.686, 0.678, 0.666), turbidity) * 0.0074;

    float zenithAngle = acos(max(0.0, dot(up, rayDir)));
    float denom = cos(zenithAngle)
            + 0.15 * pow(93.885 - ((zenithAngle * 180.0) / MOE_PI), -1.253);
    float sR = 3795.0 / denom;
    float sM = 7100.0 / denom;
    float3 fex = exp(-(betaR * sR + betaM * sM));

    float3 sunDirection = normalize(sunPosition);
    float cosTheta = dot(rayDir, sunDirection);
    float3 betaRTheta = betaR * RayleighPhase(cosTheta * 0.5 + 0.5);
    float3 betaMTheta = betaM * HgPhase(cosTheta, 0.468);
    float sunE = SunIntensity(dot(sunDirection, up));
    float3 lin = pow(max(sunE * ((betaRTheta + betaMTheta) / (betaR + betaM))
                    * (1.0 - fex), 0.0), 1.5);
    lin *= lerp(1.0,
            pow(max(sunE * ((betaRTheta + betaMTheta) / (betaR + betaM)) * fex, 0.0), 0.5),
            saturate(pow(1.0 - dot(up, sunDirection), 5.0)));

    float3 whiteScale = 1.0 / Uncharted2Tonemap(float3(19.50));
    // Deep-blue sky baseline (the reference's 0.04 scale leaves the sky washed
    // out once tone.frag brightens it with a white point of 1.0).
    const float kSkyRadianceScale = 0.015;
    float3 base = lin * kSkyRadianceScale + float3(0.0, 0.001, 0.0025) * 0.3;
    float3 color = Uncharted2Tonemap(log2(2.0) * base) * whiteScale;
    noSun = pow(max(color, 0.0), float3(1.0 / (1.2 + 1.2 * sunfade)));

    float sunAngularDiameterCos = cos(0.032);
    sunDisk = smoothstep(sunAngularDiameterCos, sunAngularDiameterCos + 0.00002, cosTheta);
    float3 l0 = float3(0.1) * fex + sunE * 1900.0 * fex * sunDisk;
    float3 withSun = (lin + l0) * kSkyRadianceScale + float3(0.0, 0.001, 0.0025) * 0.3;
    float3 colorWithSun = Uncharted2Tonemap(log2(2.0) * withSun) * whiteScale;
    colorWithSun = pow(max(colorWithSun, 0.0), float3(1.0 / (1.2 + 1.2 * sunfade)));

    // deepen toward a saturated blue: the Preetham curve leaves the daytime sky
    // pale, which makes it hard to read the cloud against it.
    float luma = dot(colorWithSun, float3(0.299, 0.587, 0.114));
    colorWithSun = lerp(float3(luma), colorWithSun, 1.7) * 0.55;

    // y-up: the sun is up when sunDirection.y > 0, so the cloud is not darkened
    night = smoothstep(-0.15, 0.05, sunDirection.y);
    return colorWithSun;
}

// ---- main ---------------------------------------------------------------

bool IntersectBox(float3 ro, float3 rd, float3 bmin, float3 bmax, out float t0, out float t1) {
    float3 inv = 1.0 / rd;
    float3 ta = (bmin - ro) * inv;
    float3 tb = (bmax - ro) * inv;
    float3 tmin = min(ta, tb);
    float3 tmax = max(ta, tb);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 >= max(t0, 0.0);
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float2 ndc = input.mUv * 2.0 - 1.0;
    float3 rayDir = normalize(pcs.mForward
            + pcs.mRight * (ndc.x * pcs.mTanHalfFov * pcs.mAspect)
            - pcs.mUp * (ndc.y * pcs.mTanHalfFov));

    float3 sunPosition = pcs.mSunDir * 400000.0;

    float3 skyNoSun;
    float sunDisk;
    float night;
    float3 sky = SkyColor(rayDir, sunPosition, pcs.mTurbidity, skyNoSun, sunDisk, night);

    float cosAngle = dot(rayDir, pcs.mSunDir);
    float3 scatter = float3(0.0);
    float transmittance = 1.0;

    float t0;
    float t1;
    if (IntersectBox(pcs.mCameraPos, rayDir, pcs.mBoxMin, pcs.mBoxMax, t0, t1)) {
        float t = max(t0, 0.0);
        float limit = min(pcs.mFarClip, t1);
        for (int i = 0; i < 512; ++i) {
            if (t >= limit || transmittance <= pcs.mTransmittanceLimit) {
                break;
            }
            float3 world = pcs.mCameraPos + rayDir * t;
            float3 uv = ToSampleCoord(world);
            ModelingData m = SampleModeling(uv);

            float adaptive = max(1.0, sqrt(t) * 0.08);
            float stepLen = min(max(m.mSdf, adaptive), limit - t);

            if (m.mSdf < 0.0 && m.mProfile > 0.0) {
                float d = DetailDensity(t, uv, m, true) * pcs.mDensityScale;
                if (d > 0.0) {
                    float densityToSun = GetDensityToSun(uv, pcs.mSunDir, stepLen);
                    float densityLod = lightGrid.SampleLevel(linearSampler, uv, 0.0).g;
                    float3 cloudColor = LightEnergy(cosAngle, t, densityToSun, densityLod,
                            m.mProfile, skyNoSun);
                    scatter += transmittance * d * cloudColor * night;
                    transmittance *= exp(-d);
                }
            }
            t += stepLen;
        }
    }

    float3 bg = sky;
    float3 color = bg * transmittance + scatter;

    // HDR output; the composite pass applies god ray + exposure/tonemap.
    return float4(color, 1.0);
}
