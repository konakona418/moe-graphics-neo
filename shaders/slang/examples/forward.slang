// [moe("vertex", "fragment")]

// Push constants are addressed by name from C++ (renderer.GetPushConstant +
// SetPushConstant); the names below are chosen freely.
struct PushConstants {
    float4x4 mvp;
    float4x4 model;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL,
        float2 uv : TEXCOORD0) {
    VertexOutput output;
    output.position = mul(pcs.mvp, float4(position, 1.0));
    output.normal = mul((float3x3) pcs.model, normal);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 n = normalize(input.normal);
    float3 lightDir = normalize(float3(0.5, 0.8, 0.6));
    float ndl = max(dot(n, lightDir), 0.0);
    float3 color = float3(0.8, 0.4, 0.3) * (0.2 + 0.8 * ndl);
    return float4(color, 1.0);
}
