// [moe("compute")]

// Velocity divergence (the right-hand side of the pressure Poisson equation).

#include "examples/fluid2d/fluid2d_common.slang"

[[vk::binding(0, 0)]] RWTexture2D<float2> velocity;
[[vk::binding(1, 0)]] RWTexture2D<float> divergence;

[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int2 id = int2(tid.xy);

    float2 vT = velocity[ClampCoord(id + kOffT)] * ObstacleNeighbor(id, kOffT);
    float2 vB = velocity[ClampCoord(id + kOffB)] * ObstacleNeighbor(id, kOffB);
    float2 vR = velocity[ClampCoord(id + kOffR)] * ObstacleNeighbor(id, kOffR);
    float2 vL = velocity[ClampCoord(id + kOffL)] * ObstacleNeighbor(id, kOffL);

    float halfPixelSize = (1.0 / kGridSizeF) * 0.5;
    float div = halfPixelSize * (vR.x - vL.x + vT.y - vB.y);
    divergence[id] = div;
}
