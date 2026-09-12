// [moe("compute")]

// One Jacobi iteration of the pressure Poisson solve. Dispatched N times.

#include "examples/fluid2d/fluid2d_common.slang"

[[vk::binding(0, 0)]] RWTexture2D<float> pressure;
[[vk::binding(1, 0)]] RWTexture2D<float> divergence;

[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int2 id = int2(tid.xy);

    float p = pressure[id];
    float pT = lerp(p, pressure[ClampCoord(id + kOffT)], ObstacleNeighbor(id, kOffT));
    float pB = lerp(p, pressure[ClampCoord(id + kOffB)], ObstacleNeighbor(id, kOffB));
    float pR = lerp(p, pressure[ClampCoord(id + kOffR)], ObstacleNeighbor(id, kOffR));
    float pL = lerp(p, pressure[ClampCoord(id + kOffL)], ObstacleNeighbor(id, kOffL));

    float div = divergence[id] * kGridSizeF;
    pressure[id] = (pL + pR + pB + pT - div) / 4.0;
}
