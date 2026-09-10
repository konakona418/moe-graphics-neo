// [moe("vertex", "fragment")]

struct PushConstants {
    float4x4 mvp;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float4 color : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL,
        float4 color : COLOR0) {
    VertexOutput output;
    output.position = mul(pcs.mvp, float4(position, 1.0));
    output.normal = normal;
    output.color = color;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 n = normalize(input.normal);
    float3 lightDir = normalize(float3(0.4, 0.9, 0.5));
    float ndl = max(dot(n, lightDir), 0.0);
    float3 color = input.color.rgb * (0.35 + 0.65 * ndl);
    return float4(color, 1.0);
}
