// [moe("vertex")]
[shader("vertex")]
float4 vertexMain(uint vertexId : SV_VulkanVertexId) : SV_Position {
    float2 pos = float2(float((vertexId << 1) & 2), float(vertexId & 2));
    return float4(pos * 2.0 - 1.0, 0.0, 1.0);
}