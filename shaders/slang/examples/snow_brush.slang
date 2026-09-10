// [moe("compute")]

// Terrain brush: sinks heights near the player's grid position (per-vertex
// independent, so a single dispatch is race-free).

[vk::push_constant]
float4 brushParams; // x,y = brush center (grid coords), z = radius (cells), w = amount

[vk::binding(0, 0)]
RWStructuredBuffer<float> heights; // (kGridSize+1)^2 floats

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= 65 || tid.y >= 65) {
        return;
    }
    const uint index = tid.y * 65 + tid.x;
    float2 d = (float2(tid.xy) - brushParams.xy) / brushParams.z;
    float dist2 = dot(d, d);
    if (dist2 >= 1.0) {
        return;
    }
    float falloff = 1.0 - dist2;
    heights[index] -= brushParams.w * falloff;
}
