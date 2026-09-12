// [moe("vertex", "fragment")]

// Post-process matching the reference Nubis3 implementation (Frostnova
// tone.frag): screen-space god rays toward the sun, then an Uncharted-2
// tonemap with a white point of 1.0 (which is what gives the vivid look),
// then a dark vignette.

#include "examples/nubis/nubis_common.slang"

struct PushConstants {
    float2 mSunScreen;      // sun direction projected to uv
    float mSunElevation;    // sunDir.y (y-up), > 0 = above horizon
    float mGodrayEnabled;
    float mGodrayExposure;
    float mExposure;
    float mWhitePoint;
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] Texture2D<float4> sceneTex;
[[vk::binding(1, 0)]] SamplerState linearSampler;

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

float3 Tonemap(float3 x, float exposure, float invGamma, float whitePoint) {
    float3 color = Uncharted2Tonemap(exposure * x);
    color *= 1.0 / Uncharted2Tonemap(float3(whitePoint));
    return pow(max(color, 0.0), float3(invGamma));
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float2 uv = input.mUv;
    float3 scene = sceneTex.SampleLevel(linearSampler, uv, 0.0).rgb;

    float godrayAmount = saturate(pcs.mGodrayEnabled);
    if (godrayAmount > 0.0 && pcs.mSunElevation > 0.0) {
        const int kSamples = 100;
        const float kDecay = 0.96;
        const float kDensity = 0.2;
        const float kWeight = 0.58767;
        // strong near the horizon, subtle at high noon (matches the reference)
        float exposure = lerp(pcs.mGodrayExposure, 0.02, saturate(pcs.mSunElevation));

        float2 delta = (uv * 2.0 - 1.0 - (pcs.mSunScreen * 2.0 - 1.0))
                * kDensity / float(kSamples);
        float2 tc = uv;
        float illumination = 1.0;
        float3 ray = scene * 0.4;
        for (int i = 0; i < kSamples; ++i) {
            tc -= delta;
            float3 samp = sceneTex.SampleLevel(linearSampler, tc, 0.0).rgb * 0.4;
            ray += samp * (illumination * kWeight);
            illumination *= kDecay;
        }
        scene += ray * (exposure * godrayAmount);
    }

    float3 col = Tonemap(scene, pcs.mExposure, 1.0 / 2.2, pcs.mWhitePoint);

    float vignette = dot(uv - 0.5, uv - 0.5);
    col = lerp(col, float3(0.1, 0.05, 0.13), vignette);
    return float4(col, 1.0);
}
