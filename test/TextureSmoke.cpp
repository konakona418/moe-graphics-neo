// Texture pipeline smoke test: decode a BMP (stb_image), upload it through the
// neo Uploader, read it back from the GPU image and verify byte-exact pixels.
// Exercises DecodeTexture + UploadTexture + image barriers + readback.

#include <Neo/TextureLoader.hpp>
#include <Neo/Uploader.hpp>

#include <RHI/CommandList.hpp>
#include <RHI/PipelineCache.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(cond)                                              \
    do {                                                         \
        if (!(cond)) {                                           \
            std::fprintf(stderr, "TextureSmoke FAILED: %s (%d)\n", \
                    #cond, __LINE__);                            \
            return 1;                                            \
        }                                                        \
    } while (false)

namespace {
    std::vector<uint8_t> ReadFile(const char* path) {
        std::ifstream file(path, std::ios::binary);
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
    }
}// namespace

int main() {
    const std::vector<uint8_t> bmp = ReadFile(MOE_SOURCE_DIR "/test/assets/texture.bmp");
    CHECK(!bmp.empty());

    moe::neo::Texture texture;
    CHECK(moe::neo::DecodeTexture(bmp.data(), bmp.size(), texture, false));
    CHECK(texture.mWidth == 4 && texture.mHeight == 4);
    CHECK(texture.mChannels == 4);
    CHECK(texture.mData.size() == 4 * 4 * 4);

    // stb_image keeps BMP rows bottom-up (first row in memory = bottom of the
    // image): bottom-left blue, bottom-right white, top-left red, top-right
    // green.
    const auto pixel = [&](uint32_t x, uint32_t y) -> uint32_t {
        const size_t off = (y * 4 + x) * 4;
        return static_cast<uint32_t>(texture.mData[off])
                | static_cast<uint32_t>(texture.mData[off + 1]) << 8
                | static_cast<uint32_t>(texture.mData[off + 2]) << 16
                | static_cast<uint32_t>(texture.mData[off + 3]) << 24;
    };
    CHECK(pixel(0, 0) == 0xFFFF0000u); // blue
    CHECK(pixel(2, 0) == 0xFFFFFFFFu); // white
    CHECK(pixel(0, 3) == 0xFF0000FFu); // red
    CHECK(pixel(2, 3) == 0xFF00FF00u); // green

    // ---- upload ----
    std::string error;
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;
    CHECK(moe::rhi::Device::Create(deviceInfo, device));

    moe::neo::Uploader uploader;
    CHECK(uploader.Init(device, error));

    moe::neo::UploadedTexture gpu;
    CHECK(uploader.UploadTexture(texture, gpu, error));

    // ---- readback ----
    moe::rhi::BufferCreateInfo readbackInfo{};
    readbackInfo.mSize = texture.mData.size();
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    moe::rhi::Buffer readback;
    CHECK(device.CreateBuffer(readbackInfo, readback));

    moe::rhi::CommandList cmd;
    CHECK(device.CreateCommandList(cmd));
    cmd.Begin();
    moe::rhi::SyncInfo toRead{};
    toRead.mSrcStage = moe::rhi::PipelineStage::kFragmentShader;
    toRead.mSrcAccess = moe::rhi::Access::kShaderRead;
    toRead.mDstStage = moe::rhi::PipelineStage::kTransfer;
    toRead.mDstAccess = moe::rhi::Access::kTransferRead;
    cmd.ImageBarrier(gpu.mImage, moe::rhi::ImageLayout::kShaderReadOnly,
            moe::rhi::ImageLayout::kTransferSrc, toRead);
    cmd.CopyImageToBuffer(gpu.mImage, readback);
    cmd.End();
    CHECK(device.Submit(cmd, true));

    const auto* pixels = static_cast<const uint8_t*>(readback.Map());
    CHECK(pixels != nullptr);
    CHECK(std::memcmp(pixels, texture.mData.data(), texture.mData.size()) == 0);
    readback.Unmap();

    readback.Destroy();
    cmd.Destroy();
    gpu.Destroy();
    cache.Destroy();
    CHECK(device.WaitIdle());
    device.Destroy();

    std::printf("TextureSmoke PASS\n");
    return 0;
}
