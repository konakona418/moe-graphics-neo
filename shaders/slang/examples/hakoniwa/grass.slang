// [moe("vertex", "fragment")]

// Hakoniwa grass clipmap. One instanced draw per ring; the vertex shader
// derives each blade from SV_VulkanInstanceId (no instance buffer):
//   instance id -> (tile in this ring's grid, slot in tile)
// Ring r uses tile size base*2^r and keeps only an annulus of tiles, so each
// ring doubles the world coverage at a quarter of the density. Rings are
// camera-centred but snapped to their own tile grid, and blades are hashed from
// the *world* tile key, so blades never swim as the camera moves. Slots outside
// the annulus collapse to a degenerate triangle.
//
// The terrain height is evaluated analytically here with the SAME formula as
// Terrain.cpp's MakeTerrainMesh()/height lambda — keep the two in sync.

struct PushConstants {
    float4x4 viewProj;
    float3 cameraPos;
    float3 sunDir;
    float3 sunColor;
    float3 ambient;
    float3 fogColor;
    float fogDensity;
    float amplitude;
    float windStrength;
    float time;
    float terrainHalfSize;
    // Per-ring, re-set before each ring's draw.
    float tileSize;
    int tilesPerSide;
    int bladesPerTile;
    int innerTiles;
    int outerTiles;
    // Player displacement field.
    float2 fieldOrigin;
    float fieldSize;
    float fieldDisplace;
};

[vk::push_constant]
PushConstants pcs;

[[vk::binding(0, 0)]]
Texture2D<float4> displacementField;
[[vk::binding(1, 0)]]
SamplerState displacementSampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : TEXCOORD0;
};

uint HashUint(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float Hash01(uint seed) {
    return float(HashUint(seed) & 0x00ffffffu) / float(0x01000000u);
}

// Keep in sync with Terrain.cpp's height lambda.
float TerrainHeight(float2 p) {
    return pcs.amplitude
            * (0.5 * sin(p.x * 0.012) * cos(p.y * 0.011)
                    + 0.3 * sin(p.x * 0.03 + 1.7)
                    + 0.2 * cos(p.y * 0.025 - 0.6));
}

[shader("vertex")]
VertexOutput vertexMain(float3 position : POSITION, uint instanceId : SV_VulkanInstanceId) {
    VertexOutput output;
    output.position = float4(0.0, 0.0, 0.0, 0.0);
    output.color = float3(0.0, 0.0, 0.0);

    const uint side = uint(pcs.tilesPerSide);
    const uint bladesPerTile = uint(pcs.bladesPerTile);
    const uint slot = instanceId % bladesPerTile;
    const uint tileId = instanceId / bladesPerTile;
    const int tx = int(tileId % side) - int(side) / 2;
    const int tz = int(tileId / side) - int(side) / 2;
    const int chebyshev = max(abs(tx), abs(tz));
    if (chebyshev < pcs.innerTiles || chebyshev >= pcs.outerTiles) {
        return output;
    }

    // World tile origin: snapped to this ring's tile grid, so a tile keeps its
    // world position (and its hashed blades) regardless of where the camera is.
    const int2 camTile = int2(floor(pcs.cameraPos.xz / pcs.tileSize));
    const int2 tile = camTile + int2(tx, tz);
    const float2 tileOrigin = float2(tile) * pcs.tileSize;
    const uint2 key = uint2(tile);

    const uint seed = HashUint(key.x * 73856093u ^ key.y * 19349663u ^ slot * 83492791u);
    const float2 jitter = float2(Hash01(seed), Hash01(seed ^ 0x68bc21ebu));
    const float2 world = tileOrigin + jitter * pcs.tileSize;

    if (abs(world.x) > pcs.terrainHalfSize || abs(world.y) > pcs.terrainHalfSize) {
        return output;
    }

    const float dist = distance(world, pcs.cameraPos.xz);
    const float yaw = Hash01(seed ^ 0x1234567u) * 6.2831853;
    const float height = lerp(0.35, 0.95, Hash01(seed ^ 0xabcdefu));
    const float width = lerp(0.7, 1.0, Hash01(seed ^ 0x5555u)) * height * 0.05;
    const float baseY = TerrainHeight(world);

    // Wind: a per-blade phase bends the blade, stronger toward the tip.
    const float phase = Hash01(seed ^ 0x777u) * 6.2831853;
    const float sway = sin(pcs.time * 1.7 + phase) * 0.5 + 0.5;
    const float bend = sway * 0.35 * pcs.windStrength * position.y * position.y;
    const float2 windDir = float2(sin(phase), cos(phase));
    // A fixed per-blade lean so the field is not a grid of vertical spikes.
    const float2 leanDir = float2(sin(yaw), cos(yaw));
    const float leanAmount = Hash01(seed ^ 0x2468u) * 0.35;

    // Billboard around Y: the blade's width axis is perpendicular to the view
    // direction, so a blade is never seen edge-on.
    const float2 toCamera = normalize(pcs.cameraPos.xz - world + 1e-5);
    const float2 right = float2(-toCamera.y, toCamera.x);

    // Player displacement: sample the field at the blade's world position and
    // push the tip away, lowering it a little.
    float2 displace = float2(0.0, 0.0);
    const float2 fieldUv = (world - pcs.fieldOrigin) / pcs.fieldSize + 0.5;
    if (all(fieldUv >= 0.0) && all(fieldUv <= 1.0)) {
        displace = displacementField.SampleLevel(displacementSampler, fieldUv, 0.0).rg
                * pcs.fieldDisplace;
    }

    const float2 tipOffset = (leanDir * leanAmount + windDir * bend) * height;
    float3 worldPos;
    worldPos.xz = world + right * (position.x * width) + tipOffset * position.y
            + displace * position.y;
    worldPos.y = baseY + position.y * height - length(displace) * position.y * 0.6;
    output.position = mul(pcs.viewProj, float4(worldPos, 1.0));

    // Color: dark root to a brighter tip, per-blade hue shift, plus a tip
    // highlight and a per-blade shading normal so dense blades stay readable.
    const float variation = Hash01(seed ^ 0x999u);
    const float3 root = lerp(float3(0.03, 0.09, 0.02), float3(0.06, 0.15, 0.03), variation);
    const float3 tip = lerp(float3(0.14, 0.36, 0.07), float3(0.24, 0.50, 0.12), variation);
    float3 albedo = lerp(root, tip, position.y);
    albedo += float3(0.10, 0.15, 0.03) * smoothstep(0.6, 1.0, position.y);
    const float3 bladeNormal = normalize(
            float3(leanDir.x * 0.6, 0.7 + Hash01(seed ^ 0xbeefu) * 0.9, leanDir.y * 0.6));
    const float ndl = saturate(dot(bladeNormal, normalize(pcs.sunDir)));
    const float3 lit = albedo * (pcs.ambient + pcs.sunColor * (0.35 + 0.65 * ndl));

    const float fog = saturate(1.0 - exp(-dist * pcs.fogDensity));
    output.color = lerp(lit, pcs.fogColor, fog);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    // alpha = material mask (0.5 = grass) so post.slang can skip its outline.
    return float4(input.color, 0.5);
}
