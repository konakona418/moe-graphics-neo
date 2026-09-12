// [moe("vertex", "fragment")]

// Halftone shading variant with a paper-white specular highlight (used by the
// torus to demonstrate per-material programs): the highlight thins the shadow
// screen out, like ink pulled away from the light.

struct PushConstants {
    float4x4 mvp;
    float4x4 model;
    float4 baseColorFactor; // paper color of the lit surface
    float3 cameraPos;
    float halftoneScale;    // world-space: lattice cells per world unit
    float halftoneAngle;    // screen angle in radians
    float shadowEdge;       // diffuse term below which the shadow screen starts
    float shadowSoftness;   // width of the shadow terminator
    float lineWidth;        // grid line width in pixels (uniform in the shadow)
    float inkStrength;      // how dark the lines get
    float screenSpace;      // 1 = pattern pinned to the screen, 0 = world
    float screenCellPixels; // screen-space: cell size in pixels
    float specStrength;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL) {
    VertexOutput output;
    output.position = mul(pcs.mvp, float4(position, 1.0));
    output.normal = mul((float3x3) pcs.model, normal);
    output.worldPos = mul(pcs.model, float4(position, 1.0)).xyz;
    return output;
}

// World-space graph-paper grid; see halftone.slang for the rationale.
float WorldGrid(float3 worldPos, float3 normal, float shadow) {
    const float ca = cos(pcs.halftoneAngle);
    const float sa = sin(pcs.halftoneAngle);
    const float3 p0 = worldPos * pcs.halftoneScale;
    const float3 p = float3(ca * p0.x - sa * p0.z, p0.y, sa * p0.x + ca * p0.z);
    const float3 n = normalize(normal);

    const float3 d = abs(p - round(p));
    const float3 slope = sqrt(max(1.0 - n * n, 1e-4));
    const float lineDistance = min(min(d.x / slope.x, d.y / slope.y), d.z / slope.z);

    const float unitsPerPixel = max(fwidth(lineDistance), 1e-5);
    const float width = pcs.lineWidth * unitsPerPixel * 0.5;
    const float aa = unitsPerPixel * 0.5;
    const float lines = 1.0 - smoothstep(width - aa, width + aa, lineDistance);

    const float cellPixels = 1.0 / unitsPerPixel;
    const float average = saturate(3.0 * pcs.lineWidth / max(cellPixels, 1e-3));
    const float fade = 1.0 - smoothstep(1.5, 4.0, cellPixels);
    return lerp(lines * shadow, average * shadow, fade);
}

// Screen-space grid: pattern pinned to the screen; see halftone.slang.
float ScreenGrid(float2 pixel, float shadow) {
    const float ca = cos(pcs.halftoneAngle);
    const float sa = sin(pcs.halftoneAngle);
    const float2 p0 = pixel / max(pcs.screenCellPixels, 1.0);
    const float2 p = float2(ca * p0.x - sa * p0.y, sa * p0.x + ca * p0.y);

    const float2 d = abs(p - round(p));
    const float lineDistance = min(d.x, d.y);
    const float unitsPerPixel = 1.0 / max(pcs.screenCellPixels, 1.0);
    const float width = pcs.lineWidth * unitsPerPixel * 0.5;
    const float aa = unitsPerPixel * 0.5;
    return (1.0 - smoothstep(width - aa, width + aa, lineDistance)) * shadow;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const float3 lightDir = normalize(float3(0.55, 0.75, 0.35));
    const float3 n = normalize(input.normal);
    const float ndl = saturate(dot(n, lightDir));

    const float soft = max(pcs.shadowSoftness, 0.001);
    float shadow = smoothstep(pcs.shadowEdge - soft, pcs.shadowEdge + soft, 1.0 - ndl);

    // paper-white highlight: thin the screen out where the light hits
    const float3 viewDir = normalize(pcs.cameraPos - input.worldPos);
    const float3 halfDir = normalize(lightDir + viewDir);
    const float spec = pow(saturate(dot(n, halfDir)), 24.0);
    shadow = saturate(shadow - spec * pcs.specStrength);

    const float worldCoverage = WorldGrid(input.worldPos, n, shadow);
    const float screenCoverage = ScreenGrid(input.position.xy, shadow);
    const float ink = lerp(worldCoverage, screenCoverage, saturate(pcs.screenSpace))
            * pcs.inkStrength;
    return float4(lerp(pcs.baseColorFactor.rgb, float3(0.06, 0.06, 0.07), ink), 1.0);
}
