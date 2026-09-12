// [moe("compute")]

// Player-centred grass displacement field. Each texel stores a horizontal
// bias (xy) that the grass vertex shader adds to the blade tip. Every frame
// the previous bias decays and a radial splat is added around the player, so
// grass parts away from them and springs back once they move on.

struct PushConstants {
    float2 fieldOrigin; // player position (XZ), the field centre
    float fieldSize;    // world size covered by the field
    float radius;       // splat radius (world units)
    float strength;     // splat magnitude (world units)
    float decay;        // per-frame retention of the previous bias
    float pad0;
    float pad1;
};

[vk::push_constant]
PushConstants pcs;

[[vk::binding(0, 0)]]
RWTexture2D<float4> displacement;

[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    uint width, height;
    displacement.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    const float2 uv = (float2(tid.xy) + 0.5) / float2(width, height);
    const float2 world = pcs.fieldOrigin + (uv - 0.5) * pcs.fieldSize;

    // Decay what was there, then add this frame's radial push. The first frame
    // reads an uninitialised image, so discard non-finite values.
    float2 previous = displacement[tid.xy].xy;
    float2 bias = (any(isnan(previous)) || any(isinf(previous)))
            ? float2(0.0, 0.0)
            : previous * pcs.decay;
    const float2 toTexel = world - pcs.fieldOrigin;
    const float dist = length(toTexel);
    if (dist < pcs.radius && dist > 1e-4) {
        float falloff = 1.0 - dist / pcs.radius;
        falloff *= falloff;
        bias += (toTexel / dist) * (falloff * pcs.strength);
    }

    displacement[tid.xy] = float4(bias, 0.0, 1.0);
}
