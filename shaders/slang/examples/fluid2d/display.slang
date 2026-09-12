// [moe("vertex", "fragment")]

// Fullscreen display of the 2D fluid dye field.

struct PushConstants {
    float exposure;
    float pad0;
    float pad1;
    float pad2;
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] Texture2D<float4> densityTex;
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

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float4 den = densityTex.Sample(linearSampler, input.mUv);
    // Linear HDR (the bloom composite tone maps the sum); exposure lifts the
    // bright dye cores above 1 so the bloom can pick them out.
    float3 col = den.rgb * pcs.exposure;
    float3 bg = float3(0.02, 0.03, 0.05);
    return float4(bg + col, 1.0);
}
