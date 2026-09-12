// [moe("compute")]

// Projects the velocity field onto its divergence-free part by subtracting the
// pressure gradient (where the curl appears).

#include "examples/fluid3d/fluid3d_common.slang"

[[vk::binding(0, 0)]] RWTexture3D<float> pressure;
[[vk::binding(1, 0)]] RWTexture3D<float4> velocity;

[numthreads(8, 8, 8)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int3 id = int3(tid);

    float p = pressure[id];
    float pT = lerp(p, pressure[ClampCoord(id + kOffT)], ObstacleNeighbor(id, kOffT));
    float pB = lerp(p, pressure[ClampCoord(id + kOffB)], ObstacleNeighbor(id, kOffB));
    float pR = lerp(p, pressure[ClampCoord(id + kOffR)], ObstacleNeighbor(id, kOffR));
    float pL = lerp(p, pressure[ClampCoord(id + kOffL)], ObstacleNeighbor(id, kOffL));
    float pF = lerp(p, pressure[ClampCoord(id + kOffF)], ObstacleNeighbor(id, kOffF));
    float pK = lerp(p, pressure[ClampCoord(id + kOffK)], ObstacleNeighbor(id, kOffK));

    float3 vel = velocity[id].xyz;
    float3 grad = float3(pR - pL, pT - pB, pF - pK);
    velocity[id] = float4(vel - grad, 0.0);
}
