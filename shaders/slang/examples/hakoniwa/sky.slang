// [moe("vertex", "fragment")]

// Hakoniwa sky: a fullscreen gradient + sun disc, drawn first (clearing the
// scene target) so terrain and grass render on top. Authored in HDR and tone
// mapped with the rest of the scene by post.slang. Reconstructs the view ray
// from the camera basis, like sun.slang.

struct PushConstants {
    float3 mForward;
    float3 mRight;
    float3 mUp;
    float mTanHalfFov;
    float mAspect;
    float3 mSunDir;
    float3 mSunColor;
    float3 mHorizonColor;
    float3 mZenithColor;
};

[vk::push_constant]
PushConstants pcs;

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
    float2 ndc = input.mUv * 2.0 - 1.0;
    float3 rayDir = normalize(pcs.mForward
            + pcs.mRight * (ndc.x * pcs.mTanHalfFov * pcs.mAspect)
            - pcs.mUp * (ndc.y * pcs.mTanHalfFov));

    // Vertical gradient: horizon colour at the eye line, zenith overhead. Below
    // the horizon it stays at the horizon colour so no black leaks under the
    // terrain silhouette.
    float up = saturate(rayDir.y);
    float3 sky = lerp(pcs.mHorizonColor, pcs.mZenithColor, pow(up, 0.45));

    // Sun disc + a broad glow.
    float sunCos = max(dot(rayDir, pcs.mSunDir), 0.0);
    sky += pcs.mSunColor * (pow(sunCos, 400.0) * 8.0 + pow(sunCos, 8.0) * 0.12);
    return float4(sky, 0.0);
}
