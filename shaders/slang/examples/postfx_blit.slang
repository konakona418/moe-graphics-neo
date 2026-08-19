// [moe("vertex", "fragment")]

// Composite pass VFX: noise displacement + chromatic aberration + vignette
// + scanlines. All strengths are push constant parameters (user-controlled).

struct PushConstants {
    float displace;  // noise displacement strength
    float chromatic; // chromatic aberration
    float vignette;  // vignette
    float scanline;  // scanline darkness
};

[vk::push_constant]
PushConstants pcs;

[vk::binding(0, 0)]
Texture2D sceneTexture;
[vk::binding(1, 0)]
SamplerState sceneSampler;
[vk::binding(2, 0)]
Texture2D noiseTexture;
[vk::binding(3, 0)]
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
    // noise-driven displacement (the "funny wobble" now a mild pulse)
    float2 noiseUv = input.uv * 0.15;
    float2 offset = (noiseTexture.Sample(noiseSampler, noiseUv).rg - 0.5) * pcs.displace;

    // chromatic aberration: RGB sampled at slightly different coordinates
    float2 uvR = input.uv + offset + float2(pcs.chromatic, 0.0);
    float2 uvG = input.uv + offset;
    float2 uvB = input.uv + offset - float2(pcs.chromatic, 0.0);
    float r = sceneTexture.Sample(sceneSampler, uvR).r;
    float g = sceneTexture.Sample(sceneSampler, uvG).g;
    float b = sceneTexture.Sample(sceneSampler, uvB).b;
    float3 color = float3(r, g, b);

    // vignette: darken toward the edges
    float2 d = input.uv - 0.5;
    color *= 1.0 - pcs.vignette * dot(d, d) * 4.0;

    // scanlines: horizontal stripes
    float scan = 1.0 - pcs.scanline * (0.5 + 0.5 * sin(input.uv.y * 700.0));
    color *= scan;

    return float4(color, 1.0);
}
