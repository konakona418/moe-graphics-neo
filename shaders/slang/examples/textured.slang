// [moe("vertex", "fragment")]

struct PushConstants {
    float4x4 mvp;
    float4x4 model;
};

[vk::push_constant]
PushConstants pcs;

[vk::binding(0, 0)]
Texture2D baseColor;
[vk::binding(1, 0)]
SamplerState baseColorSampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float2 uv : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL,
        float2 uv : TEXCOORD0) {
    VertexOutput output;
    output.position = mul(pcs.mvp, float4(position, 1.0));
    output.normal = mul((float3x3) pcs.model, normal);
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 n = normalize(input.normal);
    float3 lightDir = normalize(float3(0.5, 0.8, 0.6));
    float ndl = max(dot(n, lightDir), 0.0);
    float3 texColor = baseColor.Sample(baseColorSampler, input.uv).rgb;
    float3 color = texColor * (0.2 + 0.8 * ndl);
    return float4(color, 1.0);
}
