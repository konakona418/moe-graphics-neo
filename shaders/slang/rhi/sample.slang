// [moe("compute")]

[[vk::binding(0)]]
RWStructuredBuffer<uint> outputBuffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 globalId : SV_DispatchThreadID) {
    outputBuffer[globalId.x] = globalId.x * 2u + 1u;
}