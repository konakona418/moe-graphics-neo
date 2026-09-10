// Bindless descriptor indexing smoke test: creates a BindlessSet (descriptor
// indexing device feature), uploads a solid-red texture, adds it at array
// index 0, renders a fullscreen triangle sampling it through the unbounded
// u_textures[] array, and verifies the offscreen target reads back as red.
// Exercises: feature request, bindless pool/layout/set, runtime AddImage,
// unbounded-array pipeline layouts, bind + NonUniformResourceIndex sampling.

#include <Neo/Uploader.hpp>

#include <RHI/BindlessSet.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Shader.hpp>

#include <cstdio>
#include <cstring>
#include <string>

#define CHECK(cond)                                              \
    do {                                                         \
        if (!(cond)) {                                           \
            std::fprintf(stderr, "BindlessSmoke FAILED: %s (%d)\n", \
                    #cond, __LINE__);                            \
            return 1;                                            \
        }                                                        \
    } while (false)

int main() {
    std::string error;

    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;
    deviceInfo.mEnableDescriptorIndexing = true;
    CHECK(moe::rhi::Device::Create(deviceInfo, device));

    // ---- solid-red texture ----
    moe::neo::Texture texture;
    texture.mWidth = 4;
    texture.mHeight = 4;
    texture.mChannels = 4;
    texture.mData.assign(4 * 4 * 4, 0x00);
    for (uint32_t i = 0; i < texture.mData.size(); i += 4) {
        texture.mData[i] = 0xFF; // R
        texture.mData[i + 3] = 0xFF; // A
    }
    moe::neo::Uploader uploader;
    CHECK(uploader.Init(device, error));
    moe::neo::UploadedTexture gpu;
    CHECK(uploader.UploadTexture(texture, gpu, error));

    // ---- bindless set + register the image at index 0 ----
    moe::rhi::BindlessSet bindless;
    CHECK(bindless.Init(device, error));
    CHECK(bindless.IsValid());
    CHECK(bindless.GetImageCapacity() == moe::rhi::BindlessSet::kMaxImages);
    CHECK(bindless.AddImage(0, gpu.mImage));
    CHECK(bindless.AddSampler(0, gpu.mSampler)); // overwrite default @0: fine

    // ---- shader + pipeline (unbounded arrays reflect count 0) ----
    moe::rhi::Shader vert;
    moe::rhi::Shader frag;
    moe::rhi::ShaderProgram program;
    CHECK(vert.Load(MOE_SOURCE_DIR "/shaders/rhi/bindless.vert.spv", moe::rhi::ShaderStage::kVertex));
    CHECK(frag.Load(MOE_SOURCE_DIR "/shaders/rhi/bindless.frag.spv", moe::rhi::ShaderStage::kFragment));
    CHECK(program.AddShader(vert) && program.AddShader(frag));

    moe::rhi::GraphicsPipelineState state{};
    state.mProgram = &program;
    state.mTopology = moe::rhi::PrimitiveTopology::kTriangleList;
    state.mColorFormatCount = 1;
    state.mColorFormats[0] = moe::rhi::Format::kR8G8B8A8Unorm;
    state.mBlendAttachmentCount = 1;
    state.mRaster.mCullMode = moe::rhi::CullMode::kNone;
    moe::rhi::GraphicsPipeline pipeline;
    CHECK(device.GetOrCreateGraphicsPipeline(state, pipeline));

    // ---- offscreen target + render ----
    moe::rhi::ImageCreateInfo targetInfo{};
    targetInfo.mType = moe::rhi::ImageType::k2D;
    targetInfo.mWidth = 4;
    targetInfo.mHeight = 4;
    targetInfo.mDepth = 1;
    targetInfo.mFormat = moe::rhi::Format::kR8G8B8A8Unorm;
    targetInfo.mUsage = moe::rhi::ImageUsage::kColorAttachment | moe::rhi::ImageUsage::kTransferSrc;
    moe::rhi::Image target;
    CHECK(device.CreateImage(targetInfo, target));

    moe::rhi::BufferCreateInfo readbackInfo{};
    readbackInfo.mSize = 4 * 4 * 4;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    moe::rhi::Buffer readback;
    CHECK(device.CreateBuffer(readbackInfo, readback));

    moe::rhi::CommandList cmd;
    CHECK(device.CreateCommandList(cmd));
    cmd.Begin();
    const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    cmd.BeginRendering(target, clear, nullptr, 0.0f);
    cmd.BindGraphicsPipeline(pipeline);
    cmd.SetViewport(4, 4);
    bindless.Bind(cmd, pipeline, 0);
    const glm::vec4 pc(0.0f, 0.0f, 0.0f, 0.0f); // texture id 0
    cmd.SetPushConstants(pipeline, 0, sizeof(pc), &pc);
    cmd.Draw(3, 1, 0, 0);
    cmd.EndRendering();

    moe::rhi::SyncInfo toTransfer{};
    toTransfer.mSrcStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
    toTransfer.mSrcAccess = moe::rhi::Access::kColorAttachmentWrite;
    toTransfer.mDstStage = moe::rhi::PipelineStage::kTransfer;
    toTransfer.mDstAccess = moe::rhi::Access::kTransferRead;
    cmd.ImageBarrier(target, moe::rhi::ImageLayout::kColorAttachment,
            moe::rhi::ImageLayout::kTransferSrc, toTransfer);
    cmd.CopyImageToBuffer(target, readback);
    cmd.End();
    CHECK(device.Submit(cmd, true));

    const auto* pixels = static_cast<const uint8_t*>(readback.Map());
    CHECK(pixels != nullptr);
    for (size_t i = 0; i < 4 * 4 * 4; i += 4) {
        if (pixels[i] != 0xFF || pixels[i + 1] != 0x00
                || pixels[i + 2] != 0x00 || pixels[i + 3] != 0xFF) {
            std::fprintf(stderr, "[bindless] pixel %zu = %02x %02x %02x %02x\n",
                    i / 4, pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]);
            return 1;
        }
    }
    readback.Unmap();

    readback.Destroy();
    cmd.Destroy();
    target.Destroy();
    bindless.Destroy();
    gpu.Destroy();
    cache.Destroy();
    CHECK(device.WaitIdle());
    device.Destroy();

    std::printf("BindlessSmoke PASS\n");
    return 0;
}
