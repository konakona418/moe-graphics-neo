// [moe("vertex", "fragment")]

// Composites the linear HDR scene with the bloom result, tone maps the sum,
// and writes to the (sRGB) swapchain: the hardware applies the linear -> sRGB
// encode, so this stays in linear space.

struct PushConstants {
    float mIntensity;
    float mPad0;
    float mPad1;
    float mPad2;
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] Texture2D<float4> sceneTex;
[[vk::binding(1, 0)]] SamplerState linearSampler;
[[vk::binding(2, 0)]] Texture2D<float4> bloomTex;

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

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 scene = sceneTex.SampleLevel(linearSampler, input.mUv, 0).rgb;
    float3 bloom = bloomTex.SampleLevel(linearSampler, input.mUv, 0).rgb;
    float3 hdr = max(scene + bloom * pcs.mIntensity, 0.0);
    // Reinhard tone map: the bloom is part of the HDR signal, so it rolls off
    // with the scene instead of clipping.
    float3 color = hdr / (1.0 + hdr);
    return float4(color, 1.0);
}
