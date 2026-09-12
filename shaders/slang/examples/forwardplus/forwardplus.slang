// [moe("vertex", "fragment")]

// Forward+ shading pass: renders baked world-space meshes with depth, and
// shades each pixel with the lights culled for its tile by the compute pass.

struct Light {
    float3 position;
    float radius;
    float3 color;
};

struct PushConstants {
    float4x4 viewProj;
    float3 cameraPos;
    float tileSize;
    uint maxLightsPerTile;
    uint screenWidth;
    uint screenHeight;
    uint tileCountX;
    float padding;
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(1, 0)]] StructuredBuffer<uint> tileCounts;
[[vk::binding(2, 0)]] StructuredBuffer<uint> lightIndices;

struct VertexOutput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL,
        float2 uv : TEXCOORD0) {
    VertexOutput output;
    output.position = mul(pcs.viewProj, float4(position, 1.0));
    output.worldPos = position;
    output.normal = normal;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float3 n = normalize(input.normal);
    const float3 worldPos = input.worldPos;

    const uint tileX = uint(input.position.x / pcs.tileSize);
    const uint tileY = uint(input.position.y / pcs.tileSize);
    const uint tileIndex = tileY * pcs.tileCountX + tileX;

    float3 color = float3(0.02, 0.02, 0.03); // ambient
    const uint count = tileCounts[tileIndex];
    for (uint i = 0; i < count; ++i) {
        const uint li = lightIndices[tileIndex * pcs.maxLightsPerTile + i];
        const Light light = lights[li];
        const float3 toLight = light.position - worldPos;
        const float dist2 = dot(toLight, toLight);
        const float dist = sqrt(dist2);
        // glTF physical falloff (same as the old engine): exactly 0 at the
        // light's radius, matching the culling radius so tiles have no seams
        const float atten = clamp(1.0 - pow(dist / light.radius, 4.0), 0.0, 1.0) / dist2;
        const float3 l = toLight / dist;
        const float ndl = max(dot(n, l), 0.0);
        color += light.color * ndl * atten;
    }
    return float4(color, 1.0);
}
