// [moe("compute")]

// Viscous diffusion of the dye density (6-neighbour Jacobi smoothing).

#include "examples/fluid3d/fluid3d_common.slang"

struct PushConstants {
    float4 mParams; // x = deltaTime
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] RWTexture3D<float4> density;

[numthreads(8, 8, 8)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int3 id = int3(tid);
    const float dt = pcs.mParams.x;

    float obstacle = Obstacle(id);
    float4 d = density[id];
    float4 dT = density[ClampCoord(id + kOffT)] * ObstacleNeighbor(id, kOffT);
    float4 dB = density[ClampCoord(id + kOffB)] * ObstacleNeighbor(id, kOffB);
    float4 dR = density[ClampCoord(id + kOffR)] * ObstacleNeighbor(id, kOffR);
    float4 dL = density[ClampCoord(id + kOffL)] * ObstacleNeighbor(id, kOffL);
    float4 dF = density[ClampCoord(id + kOffF)] * ObstacleNeighbor(id, kOffF);
    float4 dK = density[ClampCoord(id + kOffK)] * ObstacleNeighbor(id, kOffK);

    float4 dnew = (d + kGridSizeF * dt * (dT + dB + dR + dL + dF + dK))
            / (1.0 + 6.0 * kGridSizeF * dt);
    dnew *= obstacle;
    density[id] = dnew;
}
