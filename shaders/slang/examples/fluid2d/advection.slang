// [moe("compute")]

// Semi-Lagrangian advection of density and velocity (in place, like the
// reference): trace backwards along the velocity field and resample.

#include "examples/fluid2d/fluid2d_common.slang"

struct PushConstants {
    float4 mParams; // x = deltaTime
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] RWTexture2D<float2> velocity;
[[vk::binding(1, 0)]] RWTexture2D<float4> density;

[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int2 id = int2(tid.xy);
    const float dt = pcs.mParams.x;

    float2 vel = velocity[id];
    float obstacle = Obstacle(id);

    float2 displacement = vel * dt * kGridSizeF;
    int2 previd = clamp(int2(round(float2(id) - displacement)), int2(0, 0),
            int2(kGridSize - 1, kGridSize - 1));

    float4 den = density[id];
    float4 denPrev = density[previd];
    den.a = 0.999 * denPrev.a * obstacle;
    den.rgb = BlendColor(den.rgb, denPrev.rgb, 0.8);
    density[id] = den;

    velocity[id] = 0.99 * velocity[previd] * obstacle;
}
