#include <RHI/Buffer.hpp>

#include <Core/Defer.hpp>
#include "TestSupport.hpp"
#include <Core/Error.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Shader.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Renders a solid-red fullscreen triangle into an offscreen color target via
// CommandList::BeginRendering, then exercises CopyImage (1:1, same format) and
// BlitImage (scaled down) and read-back-verifies both. Validation is on.
int main() {
    constexpr const char* kTestName = "Render smoke (copy)";
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;

    constexpr uint32_t kSize = 4;
    constexpr uint32_t kSmall = 2;
    constexpr uint32_t kPixelCount = kSize * kSize;
    constexpr uint32_t kSmallPixelCount = kSmall * kSmall;
    const float kClear[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    moe::rhi::Shader vertShader;
    moe::rhi::Shader fragShader;
    moe::rhi::ShaderProgram graphicsProgram;
    moe::rhi::GraphicsPipeline graphicsPipeline;
    moe::rhi::Image colorTarget;
    moe::rhi::Image copyDst;
    moe::rhi::Image blitDst;
    moe::rhi::Buffer readback;
    moe::rhi::Buffer readbackSmall;
    moe::rhi::CommandList commandList;
    moe::rhi::BufferCreateInfo readbackInfo{};
    moe::rhi::GraphicsPipelineState state{};
    moe::rhi::ImageCreateInfo targetInfo{};

    moe::rhi::SyncInfo toColor{};
    moe::rhi::SyncInfo colorToTransfer{};
    moe::rhi::SyncInfo toTransferDst{};
    moe::rhi::SyncInfo transferToTransfer{};


    moe::Defer cleanup([&] {
        commandList.Destroy();
        readbackSmall.Destroy();
        readback.Destroy();
        blitDst.Destroy();
        copyDst.Destroy();
        colorTarget.Destroy();
        cache.Destroy();
        device.Destroy();
    });
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }

    if (!vertShader.Load(MOE_SOURCE_DIR "/shaders/rhi/fullscreen.vert.spv", moe::rhi::ShaderStage::kVertex)
            || !fragShader.Load(MOE_SOURCE_DIR "/shaders/rhi/flat.frag.spv", moe::rhi::ShaderStage::kFragment)) {
        return moe::test::Fail(kTestName, "shader load failed");
    }
    if (!graphicsProgram.AddShader(vertShader) || !graphicsProgram.AddShader(fragShader)) {
        return moe::test::Fail(kTestName, "graphics program add failed");
    }

    state.mProgram = &graphicsProgram;
    state.mTopology = moe::rhi::PrimitiveTopology::kTriangleList;
    state.mRaster.mCullMode = moe::rhi::CullMode::kNone; // fullscreen triangle: no culling
    state.mColorFormatCount = 1;
    state.mColorFormats[0] = moe::rhi::Format::kR8G8B8A8Unorm;
    state.mBlendAttachmentCount = 1;
    if (!device.GetOrCreateGraphicsPipeline(state, graphicsPipeline)) {
        return moe::test::Fail(kTestName);
    }

    targetInfo.mType = moe::rhi::ImageType::k2D;
    targetInfo.mWidth = kSize;
    targetInfo.mHeight = kSize;
    targetInfo.mDepth = 1;
    targetInfo.mFormat = moe::rhi::Format::kR8G8B8A8Unorm;
    targetInfo.mUsage = moe::rhi::ImageUsage::kColorAttachment | moe::rhi::ImageUsage::kTransferSrc;
    if (!device.CreateImage(targetInfo, colorTarget)) {
        return moe::test::Fail(kTestName);
    }

    targetInfo.mUsage = moe::rhi::ImageUsage::kTransferDst | moe::rhi::ImageUsage::kTransferSrc;
    if (!device.CreateImage(targetInfo, copyDst)) {
        return moe::test::Fail(kTestName);
    }

    targetInfo.mWidth = kSmall;
    targetInfo.mHeight = kSmall;
    if (!device.CreateImage(targetInfo, blitDst)) {
        return moe::test::Fail(kTestName);
    }

    readbackInfo.mSize = sizeof(uint32_t) * kPixelCount;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    if (!device.CreateBuffer(readbackInfo, readback)) {
        return moe::test::Fail(kTestName);
    }
    readbackInfo.mSize = sizeof(uint32_t) * kSmallPixelCount;
    if (!device.CreateBuffer(readbackInfo, readbackSmall)) {
        return moe::test::Fail(kTestName);
    }

    if (!device.CreateCommandList(commandList)) {
        return moe::test::Fail(kTestName);
    }

    toColor.mSrcStage = moe::rhi::PipelineStage::kTopOfPipe;
    toColor.mSrcAccess = moe::rhi::Access::kNone;
    toColor.mDstStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
    toColor.mDstAccess = moe::rhi::Access::kColorAttachmentWrite;

    colorToTransfer.mSrcStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
    colorToTransfer.mSrcAccess = moe::rhi::Access::kColorAttachmentWrite;
    colorToTransfer.mDstStage = moe::rhi::PipelineStage::kTransfer;
    colorToTransfer.mDstAccess = moe::rhi::Access::kTransferRead;

    toTransferDst.mSrcStage = moe::rhi::PipelineStage::kTopOfPipe;
    toTransferDst.mSrcAccess = moe::rhi::Access::kNone;
    toTransferDst.mDstStage = moe::rhi::PipelineStage::kTransfer;
    toTransferDst.mDstAccess = moe::rhi::Access::kTransferWrite;

    transferToTransfer.mSrcStage = moe::rhi::PipelineStage::kTransfer;
    transferToTransfer.mSrcAccess = moe::rhi::Access::kTransferWrite;
    transferToTransfer.mDstStage = moe::rhi::PipelineStage::kTransfer;
    transferToTransfer.mDstAccess = moe::rhi::Access::kTransferRead;

    commandList.Begin();

    // render solid red into colorTarget
    commandList.ImageBarrier(colorTarget, moe::rhi::ImageLayout::kUndefined,
            moe::rhi::ImageLayout::kColorAttachment, toColor);
    commandList.BeginRendering(colorTarget, kClear);
    commandList.BindGraphicsPipeline(graphicsPipeline);
    commandList.SetViewport(kSize, kSize);
    commandList.Draw(3, 1, 0, 0);
    commandList.EndRendering();

    // copy 1:1 colorTarget -> copyDst
    commandList.ImageBarrier(colorTarget, moe::rhi::ImageLayout::kColorAttachment,
            moe::rhi::ImageLayout::kTransferSrc, colorToTransfer);
    commandList.ImageBarrier(copyDst, moe::rhi::ImageLayout::kUndefined,
            moe::rhi::ImageLayout::kTransferDst, toTransferDst);
    commandList.CopyImage(colorTarget, moe::rhi::ImageLayout::kTransferSrc,
            copyDst, moe::rhi::ImageLayout::kTransferDst);
    commandList.ImageBarrier(copyDst, moe::rhi::ImageLayout::kTransferDst,
            moe::rhi::ImageLayout::kTransferSrc, transferToTransfer);
    commandList.CopyImageToBuffer(copyDst, readback);

    // blit scaled 4x4 -> 2x2 (colorTarget is already TransferSrc)
    commandList.ImageBarrier(blitDst, moe::rhi::ImageLayout::kUndefined,
            moe::rhi::ImageLayout::kTransferDst, toTransferDst);
    commandList.BlitImage(colorTarget, moe::rhi::ImageLayout::kTransferSrc,
            blitDst, moe::rhi::ImageLayout::kTransferDst, moe::rhi::Filter::kLinear);
    commandList.ImageBarrier(blitDst, moe::rhi::ImageLayout::kTransferDst,
            moe::rhi::ImageLayout::kTransferSrc, transferToTransfer);
    commandList.CopyImageToBuffer(blitDst, readbackSmall);

    commandList.End();
    if (!device.Submit(commandList, true)) {
        return moe::test::Fail(kTestName);
    }

    {
        const uint32_t expected = 0xFF0000FFu; // solid red, ABGR in R8G8B8A8
        auto* data = static_cast<uint32_t*>(readback.Map());
        if (!data) {
            return moe::test::Fail(kTestName, "failed to map copy readback");
        }
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            if (data[i] != expected) {
                std::fprintf(stderr, "Render smoke (copy) FAILED: data[%u] = 0x%08X, expected 0x%08X\n",
                        i, data[i], expected);
                readback.Unmap();
                return moe::test::Fail(kTestName);
            }
        }
        readback.Unmap();
    }

    {
        const uint32_t expected = 0xFF0000FFu; // scaled red stays red
        auto* data = static_cast<uint32_t*>(readbackSmall.Map());
        if (!data) {
            return moe::test::Fail(kTestName, "failed to map blit readback");
        }
        for (uint32_t i = 0; i < kSmallPixelCount; ++i) {
            if (data[i] != expected) {
                std::fprintf(stderr, "Render smoke (blit) FAILED: data[%u] = 0x%08X, expected 0x%08X\n",
                        i, data[i], expected);
                readbackSmall.Unmap();
                return moe::test::Fail(kTestName);
            }
        }
        readbackSmall.Unmap();
    }

    std::printf("Render smoke passed.\n");

    return EXIT_SUCCESS;
}
