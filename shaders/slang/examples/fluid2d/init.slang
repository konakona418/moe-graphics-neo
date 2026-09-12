// [moe("compute")]

// Zeroes the solver state. Dispatched once before the first simulation step.

#include "examples/fluid2d/fluid2d_common.slang"

[[vk::binding(0, 0)]] RWTexture2D<float2> velocity;
[[vk::binding(1, 0)]] RWTexture2D<float4> density;
[[vk::binding(2, 0)]] RWTexture2D<float> pressure;
[[vk::binding(3, 0)]] RWTexture2D<float> divergence;

[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    velocity[tid.xy] = float2(0.0, 0.0);
    density[tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
    pressure[tid.xy] = 0.0;
    divergence[tid.xy] = 0.0;
}
