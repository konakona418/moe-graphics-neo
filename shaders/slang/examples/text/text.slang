// [moe("vertex", "fragment")]

// Text rendering from glyph outlines (Eric Lengyel, "GPU-Centered Font
// Rendering Directly from Glyph Outlines"): glyph outlines are quadratic
// Bezier curves stored in a storage buffer, bucketed into horizontal/
// vertical bands. The vertex stage dilates each glyph quad by half a pixel
// so antialiased edges are not clipped, and the fragment stage finds the
// curves near the sample, solves them for the exact coverage, and combines
// both ray directions.
//
// The glyph data layout is produced by neo::FontData:
//   curveData: 2 float4 per curve (p1, p2) and (p3, _)
//   bandData:  per glyph, 16 x-slab metas + 16 y-slab metas of
//              (dataOffset, count), then the x-slab curve indices and the
//              y-slab curve indices (offsets relative to the glyph block).
// Vertex attributes: pos.xy/normal.xy, tex.xy/curveBase/bandBase, inverse
// Jacobian, band scale/offset, straight-alpha color.

static const uint kBandSplits = 16;

struct PushConstants {
    float4x4 viewProj;
    float4x4 model;
    float2 viewport; // framebuffer size in pixels
};

[vk::push_constant]
PushConstants pcs;

[[vk::binding(0, 0)]]
StructuredBuffer<float4> curveData;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> bandData;

struct VertexOutput {
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD0;
    nointerpolation float4 banding : TEXCOORD1;
    nointerpolation uint2 glyph : TEXCOORD2;
    float4 color : COLOR0;
};

// Pushes a quad corner outward along its normal so the coverage ramp is not
// clipped at the glyph's bounding box. `texcoord` is returned in em space.
float2 DilateQuad(float4 pos, float4 tex, float4 jac, float4 m0, float4 m1, float4 m3,
        float2 dim, out float2 vpos) {
    const float2 n = normalize(pos.zw);
    const float s = dot(m3.xy, pos.xy) + m3.w;
    const float t = dot(m3.xy, n);

    const float u = (s * dot(m0.xy, n) - t * (dot(m0.xy, pos.xy) + m0.w)) * dim.x;
    const float v = (s * dot(m1.xy, n) - t * (dot(m1.xy, pos.xy) + m1.w)) * dim.y;

    const float s2 = s * s;
    const float st = s * t;
    const float uv = u * u + v * v;
    const float2 d = pos.zw * (s2 * (st + sqrt(max(uv, 0.0))) / (uv - st * st));

    vpos = pos.xy + d;
    return float2(tex.x + dot(d, jac.xy), tex.y + dot(d, jac.zw));
}

[shader("vertex")]
VertexOutput vertexMain(
        float4 pos : POSITION,
        float4 tex : TEXCOORD0,
        float4 jac : TEXCOORD1,
        float4 bnd : TEXCOORD2,
        float4 col : COLOR0) {
    const float4x4 mvp = mul(pcs.viewProj, pcs.model);
    const float4 m0 = mvp[0];
    const float4 m1 = mvp[1];
    const float4 m2 = mvp[2];
    const float4 m3 = mvp[3];

    float2 p;
    VertexOutput output;
    output.texcoord = DilateQuad(pos, tex, jac, m0, m1, m3, pcs.viewport, p);
    output.position = float4(p.x * m0.x + p.y * m0.y + m0.w,
            p.x * m1.x + p.y * m1.y + m1.w,
            p.x * m2.x + p.y * m2.y + m2.w,
            p.x * m3.x + p.y * m3.y + m3.w);
    output.banding = bnd;
    output.glyph = uint2(uint(tex.w + 0.5), uint(tex.z + 0.5));
    output.color = col;
    return output;
}

// Tells which roots of the quadratic make a contribution from the signs of
// the three control points (bits 0 and 8 of the result).
uint CalcRootCode(float y1, float y2, float y3) {
    const uint i1 = asuint(y1) >> 31U;
    const uint i2 = asuint(y2) >> 30U;
    const uint i3 = asuint(y3) >> 29U;
    uint shift = (i2 & 2U) | (i1 & ~2U);
    shift = (i3 & 4U) | (shift & ~4U);
    return ((0x2E74U >> shift) & 0x0101U);
}

// Solves for the t values where the curve crosses y = 0, then returns the x
// coordinates at those roots.
float2 SolveHorizPoly(float4 p12, float2 p3) {
    const float2 a = p12.xy - p12.zw * 2.0 + p3;
    const float2 b = p12.xy - p12.zw;
    const float ra = 1.0 / a.y;
    const float rb = 0.5 / b.y;
    const float d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    float t1 = (b.y - d) * ra;
    float t2 = (b.y + d) * ra;
    if (abs(a.y) < 1.0 / 65536.0) {
        t1 = t2 = p12.y * rb;
    }
    return float2((a.x * t1 - b.x * 2.0) * t1 + p12.x, (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}

// Same for the crossings of x = 0; returns the y coordinates at the roots.
float2 SolveVertPoly(float4 p12, float2 p3) {
    const float2 a = p12.xy - p12.zw * 2.0 + p3;
    const float2 b = p12.xy - p12.zw;
    const float ra = 1.0 / a.x;
    const float rb = 0.5 / b.x;
    const float d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    float t1 = (b.x - d) * ra;
    float t2 = (b.x + d) * ra;
    if (abs(a.x) < 1.0 / 65536.0) {
        t1 = t2 = p12.x * rb;
    }
    return float2((a.y * t1 - b.y * 2.0) * t1 + p12.y, (a.y * t2 - b.y * 2.0) * t2 + p12.y);
}

// Combines the horizontal and vertical ray coverages, weighted by how close
// the corresponding roots are to the sample.
float CalcCoverage(float xcov, float ycov, float xwgt, float ywgt) {
    const float coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
            min(abs(xcov), abs(ycov)));
    return saturate(coverage);
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float2 emsPerPixel = fwidth(input.texcoord);
    const float2 pixelsPerEm = 1.0 / emsPerPixel;

    const int2 bandMax = int2(kBandSplits - 1, kBandSplits - 1);
    const int2 bandIndex =
            clamp(int2(input.texcoord * input.banding.xy + input.banding.zw), int2(0, 0), bandMax);
    const uint bandBase = input.glyph.x;
    const uint curveBase = input.glyph.y;

    float xcov = 0.0;
    float xwgt = 0.0;

    // Horizontal bands (y slabs) drive the horizontal ray.
    const uint hMeta = bandBase + (kBandSplits + uint(bandIndex.y)) * 2;
    const uint hCount = bandData[hMeta + 1];
    const uint hData = bandBase + bandData[hMeta];
    for (uint i = 0; i < hCount; ++i) {
        const uint curveIndex = curveBase + bandData[hData + i];
        const float4 p12 = curveData[curveIndex * 2] - float4(input.texcoord, input.texcoord);
        const float2 p3 = curveData[curveIndex * 2 + 1].xy - input.texcoord;
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) {
            break;
        }
        const uint code = CalcRootCode(p12.y, p12.w, p3.y);
        if (code != 0) {
            const float2 r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;
            if ((code & 1U) != 0U) {
                xcov += saturate(r.x + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.x) * 2.0));
            }
            if (code > 1U) {
                xcov -= saturate(r.y + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    float ycov = 0.0;
    float ywgt = 0.0;

    // Vertical bands (x slabs) drive the vertical ray.
    const uint vMeta = bandBase + uint(bandIndex.x) * 2;
    const uint vCount = bandData[vMeta + 1];
    const uint vData = bandBase + bandData[vMeta];
    for (uint i = 0; i < vCount; ++i) {
        const uint curveIndex = curveBase + bandData[vData + i];
        const float4 p12 = curveData[curveIndex * 2] - float4(input.texcoord, input.texcoord);
        const float2 p3 = curveData[curveIndex * 2 + 1].xy - input.texcoord;
        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) {
            break;
        }
        const uint code = CalcRootCode(p12.x, p12.z, p3.x);
        if (code != 0) {
            const float2 r = SolveVertPoly(p12, p3) * pixelsPerEm.y;
            if ((code & 1U) != 0U) {
                ycov -= saturate(r.x + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.x) * 2.0));
            }
            if (code > 1U) {
                ycov += saturate(r.y + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    // Premultiplied coverage (blend with src = One, dst = OneMinusSrcAlpha).
    return input.color * CalcCoverage(xcov, ycov, xwgt, ywgt);
}
