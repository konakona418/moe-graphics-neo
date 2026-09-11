// [moe("vertex", "fragment")]

// Toon-shaded volumetric clouds. Raymarches the Nubis NVDF (profile/type/
// density/SDF), erodes it with the detail-noise volume and lights it through
// the precomputed light grid. Unlike the photoreal Nubis pass, the single
// scattering is quantized into flat tone bands (cel shading) and the shadowed
// cloud is overprinted with a screen-space halftone dot lattice — the same
// pixel-space coordinate system used by the halftone demo. Single full-res
// pass straight into the swapchain.

#include "examples/nubis_common.slang"

struct PushConstants {
    float3 mCameraPos;
    float3 mForward;
    float3 mRight;
    float3 mUp;
    float3 mSunDir;
    float mTanHalfFov;
    float mAspect;
    float3 mBoxMin;
    float3 mBoxMax;
    float mDensityScale;
    float mTilingFreq;         // detail-noise tiling frequency
    float mFarClip;            // march distance limit
    float mTransmittanceLimit; // early-out transmittance
    float mSteps;              // number of flat toon tone bands
    float mShadowEdge;         // band value below which the halftone screen starts
    float mRimStrength;        // gold/silver lining on the translucent edges
    float mHalftoneCell;       // halftone cell size in pixels
    float mHalftoneAngle;      // halftone screen angle in radians
    float mInkStrength;        // how dark the halftone ink gets
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

// Nubis detail erosion (identical to the photoreal pass): blend wispy/billow
// noise by the detail type, then erode the profile and modulate by density.
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

// Sun colour as the sun moves from the horizon (dawn/dusk) to the zenith:
// deep gold near the horizon, amber in between, neutral white high up. The
// cloud rim picks this up, so the lining is gold at dawn/dusk and silver at
// noon.
float3 SunColor(float sunY) {
    float t = saturate((sunY + 0.08) / 0.7);
    float3 low = float3(1.0, 0.45, 0.18);   // red-gold at the horizon
    float3 mid = float3(1.0, 0.72, 0.40);   // amber
    float3 high = float3(1.0, 0.97, 0.90);  // near-white at noon
    return t < 0.5 ? lerp(low, mid, t * 2.0) : lerp(mid, high, (t - 0.5) * 2.0);
}

// Cel-shaded single scattering. Beer-Lambert transmittance toward the sun is
// the direct-light term; it is quantized into `mSteps` flat tone blocks, then
// tinted cool in shadow and with the time-of-day sun colour in light. `band`
// reports the quantized tone (0 = deepest shadow, 1 = fully lit) so the
// halftone screen can be gated by it.
float3 ToonLightEnergy(float cosAngle, float densityToSun, float3 skyNoSun,
        out float band) {
    float sunT = exp(-densityToSun);

    // flat tone blocks (cel shading): quantize the sun transmittance itself so
    // the shadow terminator lands as hard bands instead of a smooth gradient
    float steps = max(pcs.mSteps, 1.0);
    band = steps <= 1.0 ? sunT : saturate(floor(sunT * steps) / (steps - 1.0));

    float3 sunTint = SunColor(pcs.mSunDir.y);
    float3 shadowTint = lerp(skyNoSun, float3(0.30, 0.40, 0.56),
            saturate(pcs.mSunDir.y + 0.5)) * 0.7;
    float3 cloudColor = lerp(shadowTint, sunTint, band);

    // forward-scatter silver lining on the sun-facing side
    float phase = max(HgPhase(cosAngle, 0.75), 0.6 * HgPhase(cosAngle, -0.3));
    cloudColor += sunTint * phase * 0.15 * sunT;

    // multiple scattering lifts the deep interior off pure black
    cloudColor += skyNoSun * (1.0 - sunT) * 0.10;
    return cloudColor;
}

// Stylized sky: a soft two-stop gradient with a gentle cel band, a warm glow
// toward the sun and a flat sun disk. The palette follows the time of day —
// deep blue at noon, orange near the horizon at dawn/dusk. `noSun` is the sky
// without the disk, used to tint the cloud shadow.
float3 ToonSky(float3 rayDir, out float3 noSun, out float sunDisk) {
    const float sunY = pcs.mSunDir.y;
    const float3 sunCol = SunColor(sunY);
    const float dusk = 1.0 - smoothstep(0.02, 0.5, sunY);

    const float3 dayHorizon = float3(0.62, 0.78, 0.95);
    const float3 dayZenith = float3(0.16, 0.42, 0.86);
    const float3 duskHorizon = float3(1.0, 0.50, 0.26);
    const float3 duskZenith = float3(0.16, 0.24, 0.52);
    const float3 horizon = lerp(dayHorizon, duskHorizon, dusk);
    const float3 zenith = lerp(dayZenith, duskZenith, dusk);

    float t = pow(saturate(rayDir.y), 0.5);
    float3 sky = lerp(horizon, zenith, t);

    // mild banding so the sky reads as a painted backdrop
    float band = floor(saturate(rayDir.y) * 5.0) / 5.0;
    sky = lerp(sky, lerp(horizon, zenith, pow(band, 0.5)), 0.35);

    // warm glow around the sun, strongest near the horizon at dawn/dusk
    const float2 rd = normalize(float2(rayDir.x, rayDir.z) + 1e-5);
    const float2 sd = normalize(float2(pcs.mSunDir.x, pcs.mSunDir.z) + 1e-5);
    float towardSun = pow(saturate(dot(rd, sd)), 3.0);
    sky += sunCol * towardSun * dusk * (1.0 - saturate(rayDir.y)) * 0.6;

    noSun = sky;

    float sunCos = dot(rayDir, pcs.mSunDir);
    sunDisk = smoothstep(cos(0.035), cos(0.02), sunCos);
    sky += sunCol * sunDisk * 1.8;
    return sky;
}

// Screen-space halftone: a rotated dot lattice in the pixel coordinate system
// (as in the halftone demo), with a pixel-consistent anti-aliased edge. `tone`
// is 0 for bare paper and 1 for a fully inked shadow.
float ScreenHalftone(float2 pixel, float tone) {
    const float ca = cos(pcs.mHalftoneAngle);
    const float sa = sin(pcs.mHalftoneAngle);
    const float2 p0 = pixel / max(pcs.mHalftoneCell, 1.0);
    const float2 p = float2(ca * p0.x - sa * p0.y, sa * p0.x + ca * p0.y);

    const float2 d = frac(p) - 0.5;
    const float dist = length(d);

    // dot radius grows with tone (classic halftone coverage)
    const float radius = 0.5 * sqrt(saturate(tone));
    const float aa = max(fwidth(dist), 1e-4);
    return 1.0 - smoothstep(radius - aa, radius + aa, dist);
}

bool IntersectBox(float3 ro, float3 rd, float3 bmin, float3 bmax, out float t0,
        out float t1) {
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

    float3 skyNoSun;
    float sunDisk;
    float3 sky = ToonSky(rayDir, skyNoSun, sunDisk);

    float cosAngle = dot(rayDir, pcs.mSunDir);
    float3 scatter = float3(0.0);
    float transmittance = 1.0;
    float bandAccum = 0.0;
    float weightAccum = 0.0;

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
                    float band;
                    float3 cloudColor =
                            ToonLightEnergy(cosAngle, densityToSun, skyNoSun, band);
                    float w = transmittance * d;
                    scatter += w * cloudColor;
                    bandAccum += w * band;
                    weightAccum += w;
                    transmittance *= exp(-d);
                }
            }
            t += stepLen;
        }
    }

    float coverage = 1.0 - transmittance;
    float avgBand = bandAccum / max(weightAccum, 1e-4);

    // gold/silver lining: the translucent cloud edges facing the sun glow with
    // the sun's own colour. The term is deliberately blown out (a sharp edge
    // plus a wide halo) so the rim reads dramatically — gold at dawn/dusk,
    // silver at noon.
    float edge = saturate(4.0 * coverage * (1.0 - coverage));
    float wide = pow(edge, 0.4);
    float facing = pow(saturate(0.5 + 0.5 * cosAngle), 2.0);
    float3 rim = SunColor(pcs.mSunDir.y) * pcs.mRimStrength * facing * (edge + 0.5 * wide);

    float3 hdr = sky * transmittance + scatter + rim;

    // tonemap to display space first so the halftone ink stays crisp
    float3 color = Uncharted2Tonemap(hdr * 0.7) / Uncharted2Tonemap(float3(1.3));

    // screen-space halftone: the shadowed cloud becomes a dot screen of the
    // shadow tone on paper. `mShadowEdge` is the tone band below which the
    // screen starts (deep shadow = solid dots, mid tones = sparse dots) and
    // `coverage` keeps the sky clean.
    float tone = saturate((pcs.mShadowEdge - avgBand) / max(pcs.mShadowEdge, 1e-3));
    float shadowTone = tone * coverage;
    float dotMask = ScreenHalftone(input.mPosition.xy, shadowTone);
    float gaps = (1.0 - dotMask) * shadowTone * pcs.mInkStrength;
    color = lerp(color, float3(0.96, 0.95, 0.92), gaps);

    return float4(color, 1.0);
}
