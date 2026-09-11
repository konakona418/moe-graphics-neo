// [moe("vertex", "fragment")]

// UI geometry: rounded rectangles, borders and textured quads. Text is drawn
// separately by text.slang through PassContext::DrawText. Solid shapes sample
// a 1x1 white texture, so one shader covers everything. Draws are batched by
// texture; the fragment outputs premultiplied alpha (matching the text shader).

struct PushConstants {
    float4x4 viewProj; // ortho or layer view-projection
    float2 offset;     // layer offset, scaled by (1 + element depth)
    float2 pad;
};

[vk::push_constant]
PushConstants pcs;

[[vk::binding(0, 0)]]
Texture2D texture0;
[[vk::binding(1, 0)]]
SamplerState sampler0;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
    float2 local : TEXCOORD1;  // position relative to the rect center
    float4 rect : TEXCOORD2;   // half size in xy
    float4 params : TEXCOORD3; // mode, radius, borderWidth, z
};

[shader("vertex")]
VertexOutput vertexMain(
        float2 pos : POSITION,
        float2 uv : TEXCOORD0,
        float4 color : COLOR0,
        float4 rect : TEXCOORD1,
        float4 params : TEXCOORD2) {
    // params.w is (1 + element depth): the offset scales with it so every
    // element moves and deeper ones move more.
    const float2 p = pos + pcs.offset * params.w;
    VertexOutput output;
    output.position = mul(pcs.viewProj, float4(p, 0.0, 1.0));
    output.uv = uv;
    output.color = color;
    // Shape-relative position: use the untranslated vertex so the rounded-rect
    // SDF follows the quad when the parallax shifts it.
    output.local = pos - rect.xy;
    output.rect = rect;
    output.params = params;
    return output;
}

float SdRoundBox(float2 p, float2 halfSize, float radius) {
    const float2 q = abs(p) - halfSize + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float4 tex = texture0.Sample(sampler0, input.uv);
    const float4 color = input.color * tex;
    const int mode = int(input.params.x + 0.5);
    if (mode == 0) {
        return float4(color.rgb * color.a, color.a);
    }
    if (mode == 3) {
        // Clip mask: keep only fragments inside the rounded rect, so the
        // stencil increment/decrement matches the shape exactly. The color is
        // irrelevant (the mask draws with a no-op blend).
        const float maskDistance = SdRoundBox(input.local, input.rect.zw, input.params.y);
        if (maskDistance > 0.0) {
            discard;
        }
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    const float d = SdRoundBox(input.local, input.rect.zw, input.params.y);
    const float fill = saturate(0.5 - d);
    float alpha = color.a * fill;
    if (mode == 2) {
        const float inner = saturate(0.5 - (d + input.params.z));
        alpha = color.a * (fill - inner);
    }
    return float4(color.rgb * alpha, alpha);
}
