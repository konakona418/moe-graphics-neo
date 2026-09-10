// [moe("vertex", "fragment")]

// Inverted-hull outline: extrude along the normal and draw with front-face
// culling (the engine's DrawModelForced + DrawState{cullMode = kFront}).

struct PushConstants {
    float4x4 mvp;
    float4x4 model;
    float outlineWidth;
    float4 outlineColor;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL) {
    VertexOutput output;
    const float3 extruded = position + normal * pcs.outlineWidth;
    output.position = mul(pcs.mvp, float4(extruded, 1.0));
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return pcs.outlineColor;
}
