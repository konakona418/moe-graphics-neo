// [moe("compute")]

struct PushConstants {
    float time;
    float pad0;
    float pad1;
    float pad2;
};

[vk::push_constant]
PushConstants pcs;

[vk::binding(0, 0)]
RWTexture2D<float4> uNoise;

[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    float2 uv = (float2(tid.xy) + 0.5) / 512.0;
    float n = sin(uv.x * 20.0 + pcs.time) * cos(uv.y * 17.0 - pcs.time * 0.7);
    n += 0.5 * sin((uv.x + uv.y) * 40.0 + pcs.time * 1.3);
    float v = n * 0.5 + 0.5;
    uNoise[tid.xy] = float4(v, v * 0.8, v * 0.6, 1.0);
}
