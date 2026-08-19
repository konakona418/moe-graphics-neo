// [moe("vertex", "fragment")]

// Ground plane: a large quad with repeating UVs, drawn as a wireframe-ish
// grid so the scene has visual structure.

struct PushConstants {
    float4x4 mvp;
};

[vk::push_constant]
PushConstants pcs;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(
        float3 position : POSITION,
        float2 uv : TEXCOORD0) {
    VertexOutput output;
    output.position = mul(pcs.mvp, float4(position, 1.0));
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    // grid lines: bright lines on a dark dirt base for contrast with grass
    float2 f = abs(frac(input.uv) - 0.5);
    float line = 1.0 - smoothstep(0.46, 0.5, max(f.x, f.y));
    float3 dirt = float3(0.16, 0.12, 0.09);
    float3 grid = float3(0.38, 0.3, 0.2);
    return float4(lerp(dirt, grid, line), 1.0);
}
