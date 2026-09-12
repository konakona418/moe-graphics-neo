// [moe("vertex", "fragment")]

// Outline prepass: view-space normal in rgb and linear view depth in a. The
// screen-space ink pass finds silhouettes (depth jumps) and creases (normal
// turns) in this buffer. Rendered with DrawModelForced into a 1x target, so
// the discontinuities stay crisp.

struct PushConstants {
    float4x4 model;
    float4x4 view;
    float4x4 viewProj;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float depth : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL) {
    VertexOutput output;
    const float4 world = mul(pcs.model, float4(position, 1.0));
    const float4 viewPos = mul(pcs.view, world);
    output.position = mul(pcs.viewProj, world);
    output.normal = mul((float3x3) pcs.view, mul((float3x3) pcs.model, normal));
    output.depth = -viewPos.z;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return float4(normalize(input.normal) * 0.5 + 0.5, input.depth);
}
