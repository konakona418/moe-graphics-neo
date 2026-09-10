// [moe("vertex", "fragment")]

// Mœbius composite: overlays the outline over the scene with a noise-driven
// screen-space jitter (hand-drawn "boiling" ink). The outline pass already
// depth-tested the hull against the scene depth (it borrowed that depth
// attachment), so this pass only distorts the outline sampling coordinates
// and blends by the outline coverage (alpha).

struct PushConstants {
    float2 resolution;
    float amplitude;  // jitter in pixels
    float noiseFrame; // 0..3: which tile of the 2x2 noise atlas
    float noiseScale; // screen-space noise frequency
};

[vk::push_constant]
PushConstants pcs;

[vk::binding(0, 0)]
Texture2D sceneColor;
[vk::binding(1, 0)]
SamplerState sceneColorSampler;
[vk::binding(2, 0)]
Texture2D outlineColor;
[vk::binding(3, 0)]
SamplerState outlineColorSampler;
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

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float3 scene = sceneColor.Sample(sceneColorSampler, input.uv).rgb;

    // 4-frame noise atlas (2x2 tiles); switching frames at animation rate
    // ("on twos") is what reads as boiling ink instead of a heat haze
    const float2 tile = float2(fmod(pcs.noiseFrame, 2.0), floor(pcs.noiseFrame * 0.5));
    const float2 noiseUv = frac(input.uv * pcs.noiseScale) * 0.5 + tile * 0.5;
    const float2 jitter = (noiseTexture.Sample(noiseSampler, noiseUv).rg - 0.5) * 2.0;
    const float2 uvJittered = input.uv + jitter * pcs.amplitude / pcs.resolution;

    const float4 outline = outlineColor.Sample(outlineColorSampler, uvJittered);
    return float4(lerp(scene, outline.rgb, outline.a), 1.0);
}
