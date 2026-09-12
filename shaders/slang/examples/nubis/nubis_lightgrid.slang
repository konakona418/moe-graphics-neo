// [moe("compute")]

// Precomputes the Nubis3 light grid: for every voxel, the density accumulated
// along the sun direction (R) plus the local profile density (G). This replaces
// the per-step secondary march toward the sun in the raymarcher.
//
// Grid covers the same world box as the NVDF volume, at lower resolution.

#include "examples/nubis/nubis_common.slang"

[[vk::binding(0, 0)]] RWTexture3D<float4> gridOut;
[[vk::binding(1, 0)]] Texture3D<float4> nvdf;
[[vk::binding(2, 0)]] SamplerState nvdfSampler;

struct LightGridPushConstants {
    float3 mBoxMin;
    float3 mBoxMax;
    float3 mGridSize;
    float3 mSunDir;
    float mPad0;
};

[[vk::push_constant]] LightGridPushConstants pcs;

float ProfileDensity(float3 uv) {
    if (any(uv < 0.0) || any(uv > 1.0)) {
        return 0.0;
    }
    float4 n = nvdf.SampleLevel(nvdfSampler, uv, 0.0);
    return n.x * n.z;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 id : SV_DispatchThreadID) {
    uint3 dims = uint3(pcs.mGridSize);
    if (any(id >= dims)) {
        return;
    }

    float3 coord = float3(id);
    float3 uv = (coord + 0.5) / float3(dims);

    float density = ProfileDensity(uv);
    float4 result = float4(0.0, density, 0.0, 0.0);
    if (density <= 0.0) {
        gridOut[id] = result;
        return;
    }

    // march one voxel at a time along the sun, in light-grid voxel space
    float3 extent = pcs.mBoxMax - pcs.mBoxMin;
    float3 voxelSize = extent / float3(dims);
    float3 dir = normalize(pcs.mSunDir / voxelSize);
    float3 p = coord;
    for (int i = 0; i < 512; ++i) {
        p += dir;
        if (any(p < 0.0) || any(p >= float3(dims))) {
            break;
        }
        density += ProfileDensity((p + 0.5) / float3(dims));
    }

    // The grid accumulates raw per-voxel profile*densityScale. Convert it to
    // optical-depth-ish units so the raymarcher's exp(-densityToSun) self-shadow
    // term varies across the cloud instead of saturating to zero everywhere.
    const float kOpticalDepthScale = 0.25;
    result.r = density * kOpticalDepthScale;
    gridOut[id] = result;
}
