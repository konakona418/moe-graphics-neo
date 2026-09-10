// [moe("vertex", "fragment")]

// Mœbius-style toon shading: flat tone blocks, cool shadows, a soft rim, and
// a weak stylized specular (the _spec variant adds the highlight).

struct PushConstants {
    float4x4 mvp;
    float4x4 model;
    float4 baseColorFactor;
    float3 cameraPos;
    float steps;
    float rimStrength;
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

float3 ToonShade(VertexOutput input) {
    const float3 lightDir = normalize(float3(0.55, 0.75, 0.35));
    const float3 n = normalize(input.normal);
    const float ndl = saturate(dot(n, lightDir));

    // quantized ramp: `steps` flat tone blocks (Mœbius flat color fields)
    const float steps = max(pcs.steps, 1.0);
    const float band = steps <= 1.0 ? 1.0 : saturate(floor(ndl * steps) / (steps - 1.0));

    const float3 base = pcs.baseColorFactor.rgb;
    const float3 shadow = base * float3(0.55, 0.60, 0.78); // cool paper shadow
    float3 color = lerp(shadow, base, band);

    // rim: warm ink-adjacent light
    const float3 viewDir = normalize(pcs.cameraPos - input.worldPos);
    const float rim = pow(1.0 - saturate(dot(n, viewDir)), 3.0) * pcs.rimStrength;
    return color + rim * float3(1.0, 0.95, 0.85);
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return float4(ToonShade(input), 1.0);
}
