// [moe("vertex", "fragment")]

// Hakoniwa terrain: a procedural heightfield lit by the shared sun. The NPR
// look (tone bands, halftone, outline) is applied later by post.slang, so this
// pass only produces a smooth lit color + depth.

struct PushConstants {
    float4x4 viewProj;
    float4x4 model;
    float3 cameraPos;
    float3 sunDir;
    float3 sunColor;
    float3 ambient;
    float3 fogColor;
    float fogDensity;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float3 normal : NORMAL,
        float2 uv : TEXCOORD0) {
    VertexOutput output;
    float4 world = mul(pcs.model, float4(position, 1.0));
    output.worldPos = world.xyz;
    output.normal = normal;
    output.uv = uv;
    output.position = mul(pcs.viewProj, world);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 n = normalize(input.normal);
    float ndl = saturate(dot(n, normalize(pcs.sunDir)));

    // Height bands: an earthy soil/moss ground so the vivid green grass blades
    // read against it, rock on the slopes, a hint of snow on the peaks.
    float h = input.worldPos.y;
    float slope = 1.0 - saturate(n.y);
    float groundNoise = 0.5 + 0.5 * sin(input.worldPos.x * 0.07) * cos(input.worldPos.z * 0.06);
    float3 soil = float3(0.30, 0.24, 0.16);
    float3 moss = float3(0.19, 0.27, 0.13);
    float3 rock = float3(0.34, 0.31, 0.28);
    float3 snow = float3(0.86, 0.88, 0.92);
    float3 albedo = lerp(soil, moss, groundNoise);
    albedo = lerp(albedo, rock, smoothstep(0.25, 0.6, slope));
    albedo = lerp(albedo, snow, smoothstep(15.0, 19.5, h));

    float3 color = albedo * (pcs.ambient + pcs.sunColor * ndl);

    // aerial perspective toward the sky color
    float dist = length(input.worldPos - pcs.cameraPos);
    float fog = 1.0 - exp(-dist * pcs.fogDensity);
    color = lerp(color, pcs.fogColor, saturate(fog));

    return float4(color, 1.0);
}
