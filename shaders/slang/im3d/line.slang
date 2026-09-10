// [moe("vertex", "fragment", "geometry")]

import moe.im3d_common;

[vk::push_constant]
Im3dPCS pcs;

[shader("vertex")]
Im3dVertexOutput vertexMain(uint vertexIndex: SV_VulkanVertexID) {
    Im3dVertexOutput output;

    Im3dVertex inVertex = pcs.vertexBuffer[vertexIndex];
    float4 color = unpackColorLittleEndian(inVertex.color);
    color.a *= smoothstep(0.0, 1.0, inVertex.positionSize.w / AA_FACTOR);
    output.color = color;

    output.size = max(inVertex.positionSize.w, AA_FACTOR);
    output.position = mul(pcs.viewProj, float4(inVertex.positionSize.xyz, 1.0));

    return output;
}

[shader("geometry"), maxvertexcount(4)]
void geometryMain(line Im3dVertexOutput input[2],
                  inout TriangleStream<Im3dVertexOutput> triStream) {
    float4 p0 = input[0].position;
    float4 p1 = input[1].position;

    float2 viewport = pcs.viewport;
    float2 pos0 = p0.xy / p0.w;
    float2 pos1 = p1.xy / p1.w;

    float2 dir = pos0 - pos1;
    dir = normalize(float2(dir.x, dir.y * viewport.y / viewport.x));// correct for aspect ratio
    float2 tng0 = float2(-dir.y, dir.x);
    float2 tng1 = tng0 * input[1].size / viewport;
    tng0 = tng0 * input[0].size / viewport;

    Im3dVertexOutput outVertex;

    // line start
    outVertex.position = float4((pos0 - tng0) * p0.w, p0.zw);
    outVertex.edgeDistance = -input[0].size;
    outVertex.size = input[0].size;
    outVertex.color = input[0].color;
    triStream.Append(outVertex);

    outVertex.position = float4((pos0 + tng0) * p0.w, p0.zw);
    outVertex.color = input[0].color;
    outVertex.edgeDistance = input[0].size;
    outVertex.size = input[0].size;
    triStream.Append(outVertex);

    // line end
    outVertex.position = float4((pos1 - tng1) * p1.w, p1.zw);
    outVertex.edgeDistance = -input[1].size;
    outVertex.size = input[1].size;
    outVertex.color = input[1].color;
    triStream.Append(outVertex);

    outVertex.position = float4((pos1 + tng1) * p1.w, p1.zw);
    outVertex.color = input[1].color;
    outVertex.size = input[1].size;
    outVertex.edgeDistance = input[1].size;
    triStream.Append(outVertex);
}

[shader("fragment")]
float4 fragmentMain(Im3dVertexOutput input) : SV_Target {
    float4 outColor = input.color;
    float smoothing = smoothstep(
            1.0, 1.0 - (AA_FACTOR / input.size),
            length(abs(input.edgeDistance) / input.size));
    outColor.a *= smoothing;
    return outColor;
}
