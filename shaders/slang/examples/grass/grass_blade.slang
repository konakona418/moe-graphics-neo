// [moe("vertex", "fragment")]

// Grass blades: instanced (float4x4 per instance, locations 1..4), bent by
// the noise texture sampled at the blade's world position (wind sway).

struct PushConstants {
    float4x4 mvp;
    float time;
};

[vk::push_constant]
PushConstants pcs;

[vk::binding(0, 0)]
Texture2D noiseTexture;
[vk::binding(1, 0)]
SamplerState noiseSampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : TEXCOORD0;
};

struct VertexIn {
    float3 position : POSITION;
    // instanced: location 1..4 (float4x4)
    float4x4 instanceTransform : INSTANCE_TRANSFORM;
};

[shader("vertex")]
VertexOutput vertexMain(VertexIn input) {
    float4 worldPos = mul(input.instanceTransform, float4(input.position, 1.0));
    // wind: sample the noise at the blade's world XZ, bend by height
    float n = noiseTexture.SampleLevel(noiseSampler, worldPos.xz * 0.25, 0.0).r;
    float sway = sin(pcs.time * 2.0 + n * 6.2831);
    worldPos.x += sway * input.position.y * 0.2;
    worldPos.z += cos(pcs.time * 1.6 + n * 6.2831) * input.position.y * 0.12;

    VertexOutput output;
    output.position = mul(pcs.mvp, worldPos);
    float shade = 0.7 + 0.3 * input.position.y;
    output.color = float3(0.15 * shade, 0.72 * shade, 0.18 * shade);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
