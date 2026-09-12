// [moe("vertex", "fragment")]

// Bloom upsample: 3x3 tent filter from a lower mip into the next larger one.
// Drawn with additive blending onto the already-downsampled target.

struct PushConstants {
    float2 mTexelSize; // source (lower mip) texel size
    float mRadius;
    float mPad0;
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] Texture2D<float4> srcTex;
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
    float2 t = pcs.mTexelSize * pcs.mRadius;
    float2 uv = input.mUv;
    float3 result = srcTex.SampleLevel(linearSampler, uv + t * float2(-1, -1), 0).rgb;
    result += srcTex.SampleLevel(linearSampler, uv + t * float2(0, -1), 0).rgb * 2.0;
    result += srcTex.SampleLevel(linearSampler, uv + t * float2(1, -1), 0).rgb;
    result += srcTex.SampleLevel(linearSampler, uv + t * float2(-1, 0), 0).rgb * 2.0;
    result += srcTex.SampleLevel(linearSampler, uv, 0).rgb * 4.0;
    result += srcTex.SampleLevel(linearSampler, uv + t * float2(1, 0), 0).rgb * 2.0;
    result += srcTex.SampleLevel(linearSampler, uv + t * float2(-1, 1), 0).rgb;
    result += srcTex.SampleLevel(linearSampler, uv + t * float2(0, 1), 0).rgb * 2.0;
    result += srcTex.SampleLevel(linearSampler, uv + t * float2(1, 1), 0).rgb;
    return float4(result / 16.0, 1.0);
}
