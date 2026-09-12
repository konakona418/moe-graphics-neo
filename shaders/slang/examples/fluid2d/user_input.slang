// [moe("compute")]

// Injects dye and momentum from the moving emitter ("sphere"), and applies a
// buoyancy-like force that pushes fluid out from the centre.

#include "examples/fluid2d/fluid2d_common.slang"

struct PushConstants {
    float4 mSphere; // xy = position (offset from centre, -0.5..0.5), zw = velocity
    float4 mParams; // x = forceIntensity, y = forceRange, z = deltaTime
    float4 mDye;    // rgb = dye colour
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] RWTexture2D<float2> velocity;
[[vk::binding(1, 0)]] RWTexture2D<float4> density;

[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    int2 id = int2(tid.xy);
    float2 uv = (float2(id) + 0.5) / kGridSizeF;
    float2 sphereUv = pcs.mSphere.xy + 0.5;

    float2 vel = velocity[id];
    float4 den = density[id];
    float obstacle = Obstacle(id);

    float dist = distance(uv, sphereUv);
    float falloff = 1.0 - smoothstep(0.0, pcs.mParams.y, dist);
    float impulse = pcs.mParams.x * falloff * pcs.mParams.z * obstacle;

    float speed = length(pcs.mSphere.zw);
    den.a += impulse * speed;
    den.rgb = BlendColor(den.rgb, pcs.mDye.rgb, impulse * speed);

    // Buoyancy: push outward from the centre instead of applying gravity.
    float2 cdir = uv - 0.5;
    vel += pcs.mParams.z * den.a * cdir * 2.0;

    // Momentum from the emitter.
    vel += pcs.mSphere.zw * impulse;

    velocity[id] = vel;
    density[id] = den;
}
