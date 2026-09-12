// [moe("vertex", "fragment")]

// Fullscreen display of the buffer the compute queue wrote. The graphics
// submission waits on the compute queue's timeline value before this runs.

[[vk::binding(0, 0)]]
StructuredBuffer<float> pattern;

struct VertexOutput {
    float4 mPosition : SV_Position;
    float2 mUv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VulkanVertexId) {
    VertexOutput output;
    const float2 pos = float2(float((vertexId << 1) & 2), float(vertexId & 2));
    output.mPosition = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    output.mUv = pos;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const uint kCount = 1024;
    const uint i = min(uint(input.mUv.x * float(kCount)), kCount - 1);
    const float value = pattern[i];
    const float3 low = float3(0.10, 0.20, 0.45);
    const float3 high = float3(1.00, 0.65, 0.20);
    return float4(lerp(low, high, value), 1.0);
}
