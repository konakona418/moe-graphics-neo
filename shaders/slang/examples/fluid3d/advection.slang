// [moe("compute")]

// Semi-Lagrangian advection of density and velocity (in place, like the
// reference): trace backwards along the velocity field and resample.

#include "examples/fluid3d/fluid3d_common.slang"

struct PushConstants {
    float4 mParams; // x = deltaTime
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] RWTexture3D<float4> velocity;
[[vk::binding(1, 0)]] RWTexture3D<float4> density;

[numthreads(8, 8, 8)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int3 id = int3(tid);
    const float dt = pcs.mParams.x;

    float3 vel = velocity[id].xyz;
    float obstacle = Obstacle(id);

    float3 displacement = vel * dt * kGridSizeF;
    int3 previd = clamp(int3(round(float3(id) - displacement)), int3(0, 0, 0),
            int3(kGridSize - 1, kGridSize - 1, kGridSize - 1));

    float4 den = density[id];
    float4 denPrev = density[previd];
    den.a = 0.999 * denPrev.a * obstacle;
    den.rgb = BlendColor(den.rgb, denPrev.rgb, 0.8);
    density[id] = den;

    velocity[id] = float4(0.99 * velocity[previd].xyz * obstacle, 0.0);
}
