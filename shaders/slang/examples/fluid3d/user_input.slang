// [moe("compute")]

// Injects dye and momentum from the moving emitter ("sphere"), and applies a
// buoyancy-like force that pushes fluid out from the centre.

#include "examples/fluid3d/fluid3d_common.slang"

struct PushConstants {
    float4 mSphere;   // xyz = position (offset from centre, -0.5..0.5)
    float4 mVelocity; // xyz = emitter velocity
    float4 mParams;   // x = forceIntensity, y = forceRange, z = deltaTime
    float4 mDye;      // rgb = dye colour
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] RWTexture3D<float4> velocity;
[[vk::binding(1, 0)]] RWTexture3D<float4> density;

[numthreads(8, 8, 8)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int3 id = int3(tid);
    float3 uv = (float3(id) + 0.5) / kGridSizeF;
    float3 sphereUv = pcs.mSphere.xyz + 0.5;

    float3 vel = velocity[id].xyz;
    float4 den = density[id];
    float obstacle = Obstacle(id);

    float dist = distance(uv, sphereUv);
    float falloff = 1.0 - smoothstep(0.0, pcs.mParams.y, dist);
    float impulse = pcs.mParams.x * falloff * pcs.mParams.z * obstacle;

    float3 sphereVel = pcs.mVelocity.xyz;
    float speed = length(sphereVel);
    den.a += impulse * speed;
    den.rgb = BlendColor(den.rgb, pcs.mDye.rgb, impulse * speed);

    // Buoyancy: push outward from the centre instead of applying gravity.
    float3 cdir = uv - 0.5;
    vel += pcs.mParams.z * den.a * cdir * pcs.mParams.x;

    // Momentum from the emitter.
    vel += sphereVel * impulse * kGridSizeF * 0.5;

    velocity[id] = float4(vel, 0.0);
    density[id] = den;
}
