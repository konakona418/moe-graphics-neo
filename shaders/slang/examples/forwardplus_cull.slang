// [moe("compute")]

// Tile-based light culling (Forward+): one thread per 16x16 pixel tile,
// projects every light into screen space and appends it to the tile's light
// list if it overlaps the tile rect. Outputs are structured buffers consumed
// by the forward shading pass.

struct Light {
    float3 position;
    float radius;
    float3 color;
};

[[vk::binding(0, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> tileCounts;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> lightIndices;

struct PushConstants {
    float4x4 viewProj;
    uint screenWidth;
    uint screenHeight;
    uint tileSize;
    uint maxLightsPerTile;
    uint tileCountX;
    uint tileCount;
    uint lightCount;
    float tanHalfFov;
};

[[vk::push_constant]] PushConstants pcs;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID) {
    const uint tileIndex = id.y * pcs.tileCountX + id.x;
    if (tileIndex >= pcs.tileCount) {
        return;
    }

    const float tileLeft = float(id.x * pcs.tileSize);
    const float tileTop = float(id.y * pcs.tileSize);
    const float tileRight = min(tileLeft + float(pcs.tileSize), float(pcs.screenWidth));
    const float tileBottom = min(tileTop + float(pcs.tileSize), float(pcs.screenHeight));

    uint count = 0;
    for (uint i = 0; i < pcs.lightCount; ++i) {
        const Light light = lights[i];

        // project the light center into screen space
        const float4 clip = mul(pcs.viewProj, float4(light.position, 1.0));
        if (clip.w <= 0.0) {
            continue; // behind the camera
        }
        const float2 ndc = clip.xy / clip.w;
        // Vulkan framebuffer: y down (clip.y = +1 maps to the bottom of the
        // screen), matching the fragment's SV_Position tile computation
        const float2 screen = float2(
                (ndc.x * 0.5 + 0.5) * float(pcs.screenWidth),
                (ndc.y * 0.5 + 0.5) * float(pcs.screenHeight));
        // on-screen radius of the light sphere (must cover the full extent of
        // the light's visual influence so tile culling has no visible seams)
        const float radiusPx = light.radius * (float(pcs.screenHeight) * 0.5)
                / (clip.w * pcs.tanHalfFov) * 1.05;

        if (screen.x + radiusPx >= tileLeft && screen.x - radiusPx <= tileRight
                && screen.y + radiusPx >= tileTop && screen.y - radiusPx <= tileBottom) {
            lightIndices[tileIndex * pcs.maxLightsPerTile + count] = i;
            ++count;
            if (count >= pcs.maxLightsPerTile) {
                break;
            }
        }
    }
    tileCounts[tileIndex] = count;
}
