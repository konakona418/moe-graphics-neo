// [moe("vertex", "fragment")]

// Bloom downsample: 13-tap filter from one mip into the next (half size).

struct PushConstants {
    float2 mTexelSize; // source texel size
    float mPad0;
    float mPad1;
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
    float2 t = pcs.mTexelSize;
    float2 uv = input.mUv;
    float3 a = srcTex.SampleLevel(linearSampler, uv + t * float2(-2, -2), 0).rgb;
    float3 b = srcTex.SampleLevel(linearSampler, uv + t * float2(0, -2), 0).rgb;
    float3 c = srcTex.SampleLevel(linearSampler, uv + t * float2(2, -2), 0).rgb;
    float3 d = srcTex.SampleLevel(linearSampler, uv + t * float2(-2, 0), 0).rgb;
    float3 e = srcTex.SampleLevel(linearSampler, uv, 0).rgb;
    float3 f = srcTex.SampleLevel(linearSampler, uv + t * float2(2, 0), 0).rgb;
    float3 g = srcTex.SampleLevel(linearSampler, uv + t * float2(-2, 2), 0).rgb;
    float3 h = srcTex.SampleLevel(linearSampler, uv + t * float2(0, 2), 0).rgb;
    float3 i = srcTex.SampleLevel(linearSampler, uv + t * float2(2, 2), 0).rgb;
    float3 j = srcTex.SampleLevel(linearSampler, uv + t * float2(-1, -1), 0).rgb;
    float3 k = srcTex.SampleLevel(linearSampler, uv + t * float2(1, -1), 0).rgb;
    float3 l = srcTex.SampleLevel(linearSampler, uv + t * float2(-1, 1), 0).rgb;
    float3 m = srcTex.SampleLevel(linearSampler, uv + t * float2(1, 1), 0).rgb;
    float3 result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
    return float4(result, 1.0);
}
