// [moe("vertex", "fragment")]

// Hakoniwa NPR post pass. Reads the scene color + depth the earlier passes
// wrote and applies, in order:
//   1. toon tone quantization (flat cel bands, hue preserved)
//   2. a screen-space halftone dot lattice over the shadowed areas
//   3. an outline from depth discontinuities + luminance edges
// Fullscreen triangle; writes straight to the swapchain.

#include "examples/nubis/nubis_common.slang"

struct PushConstants {
    float2 mResolution;
    float mNearPlane;
    float mFarPlane;
    float mToonLevels;
    float mHalftoneCell;
    float mHalftoneAngle;
    float mInkStrength;
    float mOutlineStrength;
    float mOutlineThreshold;
    float mGrassOutlineSuppress;
};

[vk::push_constant]
PushConstants pcs;

[[vk::binding(0, 0)]] Texture2D<float4> sceneColor;
[[vk::binding(1, 0)]] Texture2D<float> sceneDepth;
[[vk::binding(2, 0)]] SamplerState linearSampler;

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

float Luma(float3 c) {
    return dot(c, float3(0.299, 0.587, 0.114));
}

// Depth is [0, 1] (GLM_FORCE_DEPTH_ZERO_TO_ONE, non-reversed).
float LinearizeDepth(float d) {
    return (pcs.mNearPlane * pcs.mFarPlane)
            / (pcs.mFarPlane - d * (pcs.mFarPlane - pcs.mNearPlane));
}

// Scene alpha is a material mask: 0 = sky, 0.5 = grass, 1 = terrain. Grass
// gets no ink (individual blades would otherwise be outlined into a mess).
float Grassness(float a) {
    return 1.0 - smoothstep(0.0, 0.3, abs(a - 0.5));
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float2 uv = input.mUv;
    float4 centerColor = sceneColor.Sample(linearSampler, uv);
    float3 hdr = centerColor.rgb;
    // Tone map the HDR scene once, here, so terrain and clouds share one curve.
    float3 raw = Uncharted2Tonemap(hdr * 1.15) / Uncharted2Tonemap(float3(1.3));
    float3 color = raw;
    float lum = Luma(color);

    // 1. toon bands: quantize luminance, keep the chroma. The darkest band is
    //    kept above zero so dawn/dusk shadows do not collapse to black.
    float levels = max(pcs.mToonLevels, 2.0);
    float q = max(floor(lum * levels), 1.0) / levels;
    color = color * (q / max(lum, 1e-4));

    // 2. halftone: rotate into screen space, stamp a dot lattice, darken dots
    //    where the pixel is in shadow
    float2 p = uv * pcs.mResolution;
    float ca = cos(pcs.mHalftoneAngle);
    float sa = sin(pcs.mHalftoneAngle);
    float2 rp = float2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
    float cell = max(pcs.mHalftoneCell, 1.0);
    float2 center = (floor(rp / cell) + 0.5) * cell;
    float dist = length(rp - center) / (cell * 0.5);
    float radius = saturate(1.0 - lum);
    float dotMask = smoothstep(radius, radius - 0.2, dist);
    float shadow = 1.0 - smoothstep(0.10, 0.5, lum);
    float ink = dotMask * pcs.mInkStrength * shadow;
    color = lerp(color, color * 0.25, saturate(ink));

    // 3. outline: depth discontinuity + luminance edge
    float2 texel = 1.0 / pcs.mResolution;
    float c = LinearizeDepth(sceneDepth.Sample(linearSampler, uv));
    float dl = abs(LinearizeDepth(sceneDepth.Sample(linearSampler, uv + float2(-texel.x, 0.0))) - c);
    float dr = abs(LinearizeDepth(sceneDepth.Sample(linearSampler, uv + float2(texel.x, 0.0))) - c);
    float dd = abs(LinearizeDepth(sceneDepth.Sample(linearSampler, uv + float2(0.0, -texel.y))) - c);
    float du = abs(LinearizeDepth(sceneDepth.Sample(linearSampler, uv + float2(0.0, texel.y))) - c);
    float depthEdge = max(max(dl, dr), max(dd, du)) / max(c, 1.0);

    float lc = Luma(raw);
    float4 sL = sceneColor.Sample(linearSampler, uv + float2(-texel.x, 0.0));
    float4 sR = sceneColor.Sample(linearSampler, uv + float2(texel.x, 0.0));
    float4 sD = sceneColor.Sample(linearSampler, uv + float2(0.0, -texel.y));
    float4 sU = sceneColor.Sample(linearSampler, uv + float2(0.0, texel.y));
    float ll = abs(Luma(sL.rgb) - lc);
    float lr = abs(Luma(sR.rgb) - lc);
    float ld = abs(Luma(sD.rgb) - lc);
    float lu = abs(Luma(sU.rgb) - lc);
    float lumaEdge = max(max(ll, lr), max(ld, lu));

    float outline = smoothstep(pcs.mOutlineThreshold, pcs.mOutlineThreshold * 2.5,
            depthEdge + lumaEdge);
    // Do not ink the brightest areas: the sun disc is a bright isolated spot,
    // and outlining it would turn it into a black dot.
    outline *= 1.0 - smoothstep(0.85, 1.0, lc);
    // Do not ink grass: the blades are too small, so per-blade edges read as
    // noise. The material mask (alpha) covers a pixel if it or any neighbour is
    // grass, which also clears the grass/terrain silhouette.
    float grassMask = max(max(Grassness(centerColor.a), Grassness(sL.a)),
            max(max(Grassness(sR.a), Grassness(sD.a)), Grassness(sU.a)));
    outline *= 1.0 - grassMask * pcs.mGrassOutlineSuppress;
    color = lerp(color, float3(0.05, 0.04, 0.06), outline * pcs.mOutlineStrength);

    return float4(color, 1.0);
}
