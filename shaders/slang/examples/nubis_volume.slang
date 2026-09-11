// [moe("compute")]

// Procedurally authors the Nubis3 NVDF volume. The cloud is a smooth-min blob
// of spheres displaced by FBM; the analytic signed distance is stored so the
// raymarcher can sphere-trace it, while the profile/type/density channels feed
// the Nubis density erosion.
//
//   R = dimensional profile   [0,1]
//   G = detail type           [0,1]  (0 wispy .. 1 billow)
//   B = density scale         [0,1]
//   A = signed distance       world metres (negative inside)

#include "examples/nubis_common.slang"

[[vk::binding(0, 0)]] RWTexture3D<float4> nvdfOut;

struct VolumePushConstants {
    float3 mBoxMin;
    float3 mBoxMax;
    float3 mGridSize;
    float mTime;
    float mSeed;
    float mPad0;
    float mPad1;
};

[[vk::push_constant]] VolumePushConstants pcs;

[shader("compute")]
[numthreads(4, 4, 4)]
void computeMain(uint3 id : SV_DispatchThreadID) {
    uint3 dims = uint3(pcs.mGridSize);
    if (any(id >= dims)) {
        return;
    }

    float3 uv = (float3(id) + 0.5) / float3(dims);
    float3 p = lerp(pcs.mBoxMin, pcs.mBoxMax, uv);

    // slow drift so the cloud evolves without rebuilding the whole field
    float3 wind = float3(pcs.mTime * 4.0, 0.0, pcs.mTime * 2.0);

    // base shape: a wide ring of spheres plus a domed top, fused with a smooth
    // minimum so the silhouette reads as a cumulus rather than a single ball.
    // Kept inside the volume box (x,z in +/-400, y in +/-200) with margin for
    // the displacement below, otherwise the cloud gets clipped at the box walls.
    float d = SdSphere(p, float3(-180.0, -30.0, -40.0), 125.0);
    d = SMin(d, SdSphere(p, float3(-30.0, -20.0, 130.0), 135.0), 60.0);
    d = SMin(d, SdSphere(p, float3(140.0, -30.0, 50.0), 125.0), 60.0);
    d = SMin(d, SdSphere(p, float3(60.0, -10.0, -150.0), 120.0), 60.0);
    d = SMin(d, SdSphere(p, float3(-80.0, 40.0, 0.0), 110.0), 70.0);
    d = SMin(d, SdSphere(p, float3(50.0, 45.0, -40.0), 100.0), 70.0);
    d = SMin(d, SdSphere(p, float3(-10.0, 70.0, 50.0), 80.0), 70.0);

    // cumulus: flatten the underside into a base
    d = max(d, -(p.y + 150.0));

    // cauliflower bumps: two displaced FBM octaves
    d -= (Fbm(p * 0.007 + wind * 0.001, 4) - 0.5) * 70.0;
    d -= (Fbm(p * 0.020 + 5.0, 3) - 0.5) * 25.0;

    float profile = saturate(-d / 110.0);

    // billow/wispy blend and density modulation
    float type = saturate(Fbm(p * 0.0024 + 31.0, 3));
    float densityScale = saturate(0.4 + 0.6 * Fbm(p * 0.0045 + 7.0, 3));
    densityScale *= saturate(1.25 - profile * 0.25);

    nvdfOut[id] = float4(profile, type, densityScale, d);
}
