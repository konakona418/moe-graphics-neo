// [moe("compute")]

// Velocity divergence (the right-hand side of the pressure Poisson equation).

#include "examples/fluid3d/fluid3d_common.slang"

[[vk::binding(0, 0)]] RWTexture3D<float4> velocity;
[[vk::binding(1, 0)]] RWTexture3D<float> divergence;

[numthreads(8, 8, 8)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int3 id = int3(tid);

    float3 vT = velocity[ClampCoord(id + kOffT)].xyz * ObstacleNeighbor(id, kOffT);
    float3 vB = velocity[ClampCoord(id + kOffB)].xyz * ObstacleNeighbor(id, kOffB);
    float3 vR = velocity[ClampCoord(id + kOffR)].xyz * ObstacleNeighbor(id, kOffR);
    float3 vL = velocity[ClampCoord(id + kOffL)].xyz * ObstacleNeighbor(id, kOffL);
    float3 vF = velocity[ClampCoord(id + kOffF)].xyz * ObstacleNeighbor(id, kOffF);
    float3 vK = velocity[ClampCoord(id + kOffK)].xyz * ObstacleNeighbor(id, kOffK);

    float halfPixelSize = (1.0 / kGridSizeF) * 0.5;
    float div = halfPixelSize * (vR.x - vL.x + vT.y - vB.y + vF.z - vK.z);
    divergence[id] = div;
}
