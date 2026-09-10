// [moe("vertex", "fragment")]

// Toon shading with a weak stylized specular highlight (used by the torus to
// demonstrate per-material programs).

struct PushConstants {
    float4x4 mvp;
    float4x4 model;
    float4 baseColorFactor;
    float3 cameraPos;
    float steps;
    float rimStrength;
    float specStrength;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL) {
    VertexOutput output;
    output.position = mul(pcs.mvp, float4(position, 1.0));
    output.normal = mul((float3x3) pcs.model, normal);
    output.worldPos = mul(pcs.model, float4(position, 1.0)).xyz;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float3 lightDir = normalize(float3(0.55, 0.75, 0.35));
    const float3 n = normalize(input.normal);
    const float ndl = saturate(dot(n, lightDir));

    const float steps = max(pcs.steps, 1.0);
    const float band = steps <= 1.0 ? 1.0 : saturate(floor(ndl * steps) / (steps - 1.0));

    const float3 base = pcs.baseColorFactor.rgb;
    const float3 shadow = base * float3(0.55, 0.60, 0.78);
    float3 color = lerp(shadow, base, band);

    const float3 viewDir = normalize(pcs.cameraPos - input.worldPos);
    const float rim = pow(1.0 - saturate(dot(n, viewDir)), 3.0) * pcs.rimStrength;
    color += rim * float3(1.0, 0.95, 0.85);

    // weak stylized highlight (kept tight so it reads as an ink accent)
    const float3 halfDir = normalize(lightDir + viewDir);
    const float spec = pow(saturate(dot(n, halfDir)), 16.0);
    color += spec * pcs.specStrength;

    return float4(color, 1.0);
}
