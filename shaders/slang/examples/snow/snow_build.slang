// [moe("compute")]

// Terrain rebuild: recomputes normals (central differences) and height-coded
// colors from the heights buffer, writing the interleaved vertex buffer.
// Runs after the brush dispatch (heights are settled by then).

[vk::push_constant]
float4 buildParams; // x = 1.0 snow / 0.0 sand

struct VertexData {
    float3 position;
    float3 normal;
    uint color; // rgba8
};

[vk::binding(0, 0)]
StructuredBuffer<float> heights;

[vk::binding(1, 0)]
RWStructuredBuffer<VertexData> vertices;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= 65 || tid.y >= 65) {
        return;
    }
    const uint index = tid.y * 65 + tid.x;
    const float h = heights[index];
    const float hL = tid.x > 0 ? heights[index - 1] : h;
    const float hR = tid.x < 64 ? heights[index + 1] : h;
    const float hD = tid.y > 0 ? heights[index - 65] : h;
    const float hU = tid.y < 64 ? heights[index + 65] : h;

    float3 normal = normalize(float3(hL - hR, 2.0, hD - hU));

    // height-coded color: low spots dark, crests bright
    float t = clamp((h + 1.5) / 3.0, 0.0, 1.0);
    float3 color = buildParams.x > 0.5
            ? lerp(float3(0.55, 0.68, 0.85), float3(0.95, 0.97, 1.0), t) // snow
            : lerp(float3(0.72, 0.55, 0.32), float3(0.95, 0.88, 0.68), t); // sand
    // packed so the memory byte order is [R, G, B, A] (matches the vertex
    // attribute's RGBA8_UNORM reading the buffer raw)
    uint rgba = (0xFFu << 24)
            | (uint(clamp(color.b, 0.0, 1.0) * 255.0) << 16)
            | (uint(clamp(color.g, 0.0, 1.0) * 255.0) << 8)
            | uint(clamp(color.r, 0.0, 1.0) * 255.0);

    VertexData v;
    v.position = float3((float(tid.x) - 32.0) * 0.5, h, (float(tid.y) - 32.0) * 0.5);
    v.normal = normal;
    v.color = rgba;
    vertices[index] = v;
}
