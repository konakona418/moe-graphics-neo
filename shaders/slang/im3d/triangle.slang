// [moe("vertex", "fragment")]

import moe.im3d_common;

[vk::push_constant]
Im3dPCS pcs;

[shader("vertex")]
Im3dVertexOutput vertexMain(uint vertexIndex: SV_VulkanVertexID) {
    Im3dVertexOutput output;

    Im3dVertex inVertex = pcs.vertexBuffer[vertexIndex];
    float4 color = unpackColorLittleEndian(inVertex.color);
    output.color = color;
    output.size = max(inVertex.positionSize.w, AA_FACTOR);
    output.position = mul(pcs.viewProj, float4(inVertex.positionSize.xyz, 1.0));

    return output;
}

[shader("fragment")]
float4 fragmentMain(Im3dVertexOutput input) : SV_Target {
    return input.color;
}
