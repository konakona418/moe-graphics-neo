// [moe("vertex", "fragment")]
//
// Bindless sampling: unbounded texture/sampler arrays at set 0, addressed
// through NonUniformResourceIndex (mirrors the old engine's moe/sampler.slang).

[vk::binding(0, 0)]
Texture2D u_textures[];
[vk::binding(1, 0)]
SamplerState u_samplers[];

[vk::push_constant]
float4 pc; // x = texture id

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexIndex : SV_VulkanVertexId) {
    // fullscreen triangle (no vertex buffer)
    VertexOutput output;
    float2 uv = float2(float((vertexIndex << 1) & 2), float(vertexIndex & 2));
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    uint texId = uint(pc.x);
    Texture2D tex = u_textures[NonUniformResourceIndex(texId)];
    SamplerState samp = u_samplers[NonUniformResourceIndex(1)]; // linear
    return tex.Sample(samp, input.uv);
}
