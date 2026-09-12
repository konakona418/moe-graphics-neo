// [moe("compute")]

[[vk::binding(0)]]
RWStructuredBuffer<uint> outputBuffer;

[vk::push_constant]
uint mFrame;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 globalId : SV_DispatchThreadID) {
    const uint i = globalId.x;
    outputBuffer[i] = i + mFrame;
}
