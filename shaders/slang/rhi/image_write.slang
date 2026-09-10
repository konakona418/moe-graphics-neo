// [moe("compute")]

[[vk::binding(0)]]
RWTexture2D<uint> outputImage;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint2 pixel : SV_DispatchThreadID) {
    outputImage[pixel] = pixel.x + pixel.y * 64u;
}