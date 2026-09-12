// [moe("compute")]

// Viscous diffusion of the dye density (Jacobi-style smoothing).

#include "examples/fluid2d/fluid2d_common.slang"

struct PushConstants {
    float4 mParams; // x = deltaTime
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] RWTexture2D<float4> density;

[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int2 id = int2(tid.xy);
    const float dt = pcs.mParams.x;

    float obstacle = Obstacle(id);
    float4 d = density[id];
    float4 dT = density[ClampCoord(id + kOffT)] * ObstacleNeighbor(id, kOffT);
    float4 dB = density[ClampCoord(id + kOffB)] * ObstacleNeighbor(id, kOffB);
    float4 dR = density[ClampCoord(id + kOffR)] * ObstacleNeighbor(id, kOffR);
    float4 dL = density[ClampCoord(id + kOffL)] * ObstacleNeighbor(id, kOffL);

    float4 dnew = (d + kGridSizeF * dt * (dT + dB + dR + dL)) / (1.0 + 4.0 * kGridSizeF * dt);
    dnew *= obstacle;
    density[id] = dnew;
}
