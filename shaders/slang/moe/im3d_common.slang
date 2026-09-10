static const float AA_FACTOR = 2.0;

struct Im3dVertex {
    float4 positionSize;
    uint color;
}

typedef Ptr<Im3dVertex, Access.Read> Im3dVertexBuffer;

struct Im3dPCS {
    Im3dVertexBuffer vertexBuffer;
    float4x4 viewProj;
    float2 viewport;
}

struct Im3dVertexOutput {
    noperspective float edgeDistance;
    noperspective float size;
    float4 color;

    float4 position : SV_Position;
}

// im3d stores rgba8 with MSB = r. Read little-endian, the LSB-first unpack
// yields (a, b, g, r); reverse it back to (r, g, b, a).
float4 unpackColorLittleEndian(uint packed) {
    return unpackUnorm4x8ToFloat(packed).wzyx;
}
