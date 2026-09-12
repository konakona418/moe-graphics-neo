// [moe("compute")]

// Generates the 128^3 detail-erosion volume consumed by the raymarcher.
// Channels mirror the Nubis modelling data:
//   R = low  frequency curl  (wispy)
//   G = high frequency curl  (wispy)
//   B = low  frequency alligator (billow)
//   A = high frequency alligator (billow)

#include "examples/nubis/nubis_common.slang"

[[vk::binding(0, 0)]] RWTexture3D<float4> noiseOut;

struct NoisePushConstants {
    uint mSize;
    float mSeed;
    float mPad0;
    float mPad1;
};

[[vk::push_constant]] NoisePushConstants pcs;

[shader("compute")]
[numthreads(4, 4, 4)]
void computeMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pcs.mSize)) {
        return;
    }

    float size = float(pcs.mSize);
    float3 p = (float3(id) + 0.5) / size * 8.0 + pcs.mSeed;

    float curlLow = Fbm(p, 4);
    float curlHigh = Fbm(p * 3.0 + 11.0, 4);
    float alliLow = 1.0 - Worley(p * 1.5);
    float alliHigh = 1.0 - Worley(p * 4.0 + 5.0);

    noiseOut[id] = float4(saturate(curlLow), saturate(curlHigh),
            saturate(alliLow), saturate(alliHigh));
}
