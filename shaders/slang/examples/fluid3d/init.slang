// [moe("compute")]

// Zeroes the solver state. Dispatched once before the first simulation step.

#include "examples/fluid3d/fluid3d_common.slang"

[[vk::binding(0, 0)]] RWTexture3D<float4> velocity;
[[vk::binding(1, 0)]] RWTexture3D<float4> density;
[[vk::binding(2, 0)]] RWTexture3D<float> pressure;
[[vk::binding(3, 0)]] RWTexture3D<float> divergence;

[numthreads(8, 8, 8)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    velocity[tid] = float4(0.0, 0.0, 0.0, 0.0);
    density[tid] = float4(0.0, 0.0, 0.0, 0.0);
    pressure[tid] = 0.0;
    divergence[tid] = 0.0;
}
