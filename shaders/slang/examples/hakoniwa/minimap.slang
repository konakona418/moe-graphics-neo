// [moe("vertex", "fragment")]

// Minimap contour pass: the top-down terrain pass wrote a linear orthographic
// depth; here that depth is turned into an elevation map with contour lines.
// (A straight top-down colour render of the terrain is mostly flat green, so
// the depth is the useful signal.)

struct PushConstants {
    float mCameraY;    // top-down camera height (view-space z = cameraY - worldY)
    float mNearPlane;  // orthographic near/far (depth is linear between them)
    float mFarPlane;
    float mInterval;   // contour line spacing in world units
    float mMinHeight;
    float mMaxHeight;
};

[vk::push_constant]
PushConstants pcs;

[[vk::binding(0, 0)]] Texture2D<float> sceneDepth;
[[vk::binding(1, 0)]] SamplerState linearSampler;

struct VertexOutput {
    float4 mPosition : SV_Position;
    float2 mUv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VulkanVertexId) {
    VertexOutput output;
    float2 pos = float2(float((vertexId << 1) & 2), float(vertexId & 2));
    output.mPosition = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    output.mUv = pos;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float depth = sceneDepth.Sample(linearSampler, input.mUv);
    if (depth >= 0.9999) {
        return float4(0.06f, 0.08f, 0.11f, 1.0f); // outside the terrain
    }

    const float viewZ = pcs.mNearPlane + depth * (pcs.mFarPlane - pcs.mNearPlane);
    const float worldY = pcs.mCameraY - viewZ;

    // elevation tint
    const float span = max(pcs.mMaxHeight - pcs.mMinHeight, 1.0f);
    const float t = saturate((worldY - pcs.mMinHeight) / span);
    const float3 low = float3(0.18f, 0.34f, 0.18f);
    const float3 high = float3(0.88f, 0.86f, 0.80f);
    float3 color = lerp(low, high, t);

    // contour lines at multiples of mInterval
    const float band = worldY / max(pcs.mInterval, 0.001f);
    const float f = frac(band);
    const float w = max(fwidth(band), 0.02f);
    const float line = 1.0f - smoothstep(0.0f, w * 1.5f, min(f, 1.0f - f));
    color = lerp(color, color * 0.3f, line);

    return float4(color, 1.0f);
}
