// [moe("vertex", "fragment")]

// Composites the UI render target over the swapchain. The UI target holds
// premultiplied alpha, so blend with src = One, dst = OneMinusSrcAlpha.

[[vk::binding(0, 0)]]
Texture2D<float4> uiTexture;
[[vk::binding(1, 0)]]
SamplerState uiSampler;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VulkanVertexId) {
    VertexOutput output;
    const float2 pos = float2(float((vertexId << 1) & 2), float(vertexId & 2));
    output.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    output.uv = pos;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return uiTexture.Sample(uiSampler, input.uv);
}
