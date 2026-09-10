// [moe("vertex", "fragment")]

import moe.im3d_common;

[vk::push_constant]
Im3dPCS pcs;

struct Im3dPointVertexOutput {
    noperspective float edgeDistance;
    noperspective float size;
    float4 color;

    float4 position : SV_Position;
    float pointSize : SV_PointSize;
}

[shader("vertex")]
Im3dPointVertexOutput vertexMain(uint vertexIndex: SV_VulkanVertexID) {
    Im3dPointVertexOutput output;

    Im3dVertex inVertex = pcs.vertexBuffer[vertexIndex];
    float4 color = unpackColorLittleEndian(inVertex.color);
    color.a *= smoothstep(0.0, 1.0, inVertex.positionSize.w / AA_FACTOR);
    output.color = color;
    output.size = max(inVertex.positionSize.w, AA_FACTOR);

    output.position = mul(pcs.viewProj, float4(inVertex.positionSize.xyz, 1.0));
    output.pointSize = output.size;
    return output;
}

[shader("fragment")]
float4 fragmentMain(Im3dVertexOutput input, float2 pointCoord: SV_PointCoord) : SV_Target {
    float4 outColor = input.color;
    float pointSize = input.size;
    float smoothing =
            smoothstep(0.5, 0.5 - (AA_FACTOR / pointSize),
                       length(pointCoord.xy - float2(0.5)));
    outColor.a *= smoothing;
    return outColor;
}
