// [moe("vertex", "fragment")]

// Screen-space Mœbius ink: finds silhouettes (linear-depth jumps) and creases
// (normal turns) in the normal+depth prepass, dilates the response to a
// constant pixel width, and overlays it on the resolved scene color. The
// hand-drawn wobble comes from offsetting the detection taps with a seamless
// noise field ("on twos" boiling) — the whole line shifts together instead of
// the mask being resampled, so the ink never breaks into dashes.

struct PushConstants {
    float2 resolution;
    float amplitude;      // jitter in pixels
    float noiseFrame;     // 0..3: which tile of the 2x2 noise atlas
    float noiseScale;     // screen-space noise frequency
    float lineWidth;      // ink line width in pixels
    float normalThreshold;
    float depthThreshold;
    float4 inkColor;
};

[vk::push_constant]
PushConstants pcs;

[vk::binding(0, 0)]
Texture2D sceneColor;
[vk::binding(1, 0)]
SamplerState sceneColorSampler;
[vk::binding(2, 0)]
Texture2D normalDepth;
[vk::binding(3, 0)]
SamplerState normalSampler;
[vk::binding(4, 0)]
Texture2D noiseTexture;
[vk::binding(5, 0)]
SamplerState noiseSampler;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexIndex : SV_VulkanVertexId) {
    // full-screen triangle: covers the whole viewport
    VertexOutput output;
    output.uv = float2((vertexIndex << 1) & 2, vertexIndex & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// 1px diagonal neighbourhood and 8 dilation directions.
static const float2 kNeighbours[4] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0)};
static const float2 kDirections[8] = {
    float2(1.0, 0.0), float2(0.70710678, 0.70710678), float2(0.0, 1.0),
    float2(-0.70710678, 0.70710678), float2(-1.0, 0.0), float2(-0.70710678, -0.70710678),
    float2(0.0, -1.0), float2(0.70710678, -0.70710678)};

// Ink coverage between the center sample and one comparison sample.
float EdgeAgainst(float4 center, float4 sampleValue) {
    const float centerDepth = center.a;
    const float sampleDepth = sampleValue.a;

    // silhouettes: relative linear-depth jump (background has depth 0)
    const float depthEdge = abs(centerDepth - sampleDepth)
            / max(max(centerDepth, sampleDepth), 0.001);
    float ink = smoothstep(pcs.depthThreshold, pcs.depthThreshold * 2.0, depthEdge);

    // creases: normal turn, only where both samples hit geometry
    const float both = step(0.001, centerDepth) * step(0.001, sampleDepth);
    const float3 cn = center.rgb * 2.0 - 1.0;
    const float3 sn = sampleValue.rgb * 2.0 - 1.0;
    const float normalEdge = (1.0 - saturate(dot(cn, sn))) * both;
    ink = max(ink, smoothstep(pcs.normalThreshold, pcs.normalThreshold * 2.0, normalEdge));
    return ink;
}

float InkAt(float2 uv) {
    const float2 texel = 1.0 / pcs.resolution;
    const float4 center = normalDepth.Sample(normalSampler, uv);
    float ink = 0.0;
    [unroll]
    for (int i = 0; i < 4; ++i) {
        const float4 s = normalDepth.Sample(normalSampler, uv + kNeighbours[i] * texel);
        ink = max(ink, EdgeAgainst(center, s));
    }
    // dilate: an edge within lineWidth/2 pixels of the center inks the center
    [unroll]
    for (int i = 0; i < 8; ++i) {
        const float4 s = normalDepth.Sample(normalSampler,
                uv + kDirections[i] * pcs.lineWidth * 0.5 * texel);
        ink = max(ink, EdgeAgainst(center, s));
    }
    return ink;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float3 scene = sceneColor.Sample(sceneColorSampler, input.uv).rgb;

    // seamless, temporally-coherent noise (2x2 tile atlas): the jitter is a
    // smooth field, so the detected line wobbles without tearing
    const float2 tile = float2(fmod(pcs.noiseFrame, 2.0), floor(pcs.noiseFrame * 0.5));
    const float2 noiseUv = frac(input.uv * pcs.noiseScale) * 0.5 + tile * 0.5;
    const float2 jitter = (noiseTexture.Sample(noiseSampler, noiseUv).rg - 0.5) * 2.0
            * pcs.amplitude / pcs.resolution;

    const float ink = InkAt(input.uv + jitter);
    return float4(lerp(scene, pcs.inkColor.rgb, ink * pcs.inkColor.a), 1.0);
}
