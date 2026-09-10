#include <RHI/Buffer.hpp>

#include <Core/Defer.hpp>
#include "TestSupport.hpp"
#include <Core/Error.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/PipelineCache.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    constexpr const char* kTestName = "Upload smoke";
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;

    constexpr uint32_t kSize = 4;
    constexpr uint32_t kPixelCount = kSize * kSize;

    moe::rhi::ImageCreateInfo imageInfo{};
    moe::rhi::Image image;
    moe::rhi::BufferCreateInfo stagingInfo{};
    moe::rhi::Buffer staging;
    moe::rhi::BufferCreateInfo readbackInfo{};
    moe::rhi::Buffer readback;
    moe::rhi::CommandList commandList;

    moe::rhi::SyncInfo transferIn{};
    moe::rhi::SyncInfo transferOut{};


    moe::Defer cleanup([&] {
        commandList.Destroy();
        readback.Destroy();
        staging.Destroy();
        image.Destroy();
        cache.Destroy();
        device.Destroy();
    });
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }

    imageInfo.mType = moe::rhi::ImageType::k2D;
    imageInfo.mWidth = kSize;
    imageInfo.mHeight = kSize;
    imageInfo.mDepth = 1;
    imageInfo.mFormat = moe::rhi::Format::kR32Uint;
    imageInfo.mUsage = moe::rhi::ImageUsage::kTransferDst | moe::rhi::ImageUsage::kTransferSrc;
    if (!device.CreateImage(imageInfo, image)) {
        return moe::test::Fail(kTestName);
    }

    stagingInfo.mSize = sizeof(uint32_t) * kPixelCount;
    stagingInfo.mUsage = moe::rhi::BufferUsage::kTransferSrc;
    stagingInfo.mCpuVisible = true;
    if (!device.CreateBuffer(stagingInfo, staging)) {
        return moe::test::Fail(kTestName);
    }

    readbackInfo.mSize = sizeof(uint32_t) * kPixelCount;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    if (!device.CreateBuffer(readbackInfo, readback)) {
        return moe::test::Fail(kTestName);
    }

    // initial transition: Undefined -> TransferDst (discard old contents)
    transferIn.mSrcStage = moe::rhi::PipelineStage::kTopOfPipe;
    transferIn.mSrcAccess = moe::rhi::Access::kNone;
    transferIn.mDstStage = moe::rhi::PipelineStage::kTransfer;
    transferIn.mDstAccess = moe::rhi::Access::kTransferWrite;
    // TransferDst -> TransferSrc between the two copies
    transferOut.mSrcStage = moe::rhi::PipelineStage::kTransfer;
    transferOut.mSrcAccess = moe::rhi::Access::kTransferWrite;
    transferOut.mDstStage = moe::rhi::PipelineStage::kTransfer;
    transferOut.mDstAccess = moe::rhi::Access::kTransferRead;

    {
        auto* data = static_cast<uint32_t*>(staging.Map());
        if (!data) {
            return moe::test::Fail(kTestName, "failed to map staging buffer");
        }
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            data[i] = i + 100u; // distinguishable from the verify pattern
        }
        staging.Unmap();
    }

    if (!device.CreateCommandList(commandList)) {
        return moe::test::Fail(kTestName);
    }

    commandList.Begin();
    commandList.ImageBarrier(image, moe::rhi::ImageLayout::kUndefined,
            moe::rhi::ImageLayout::kTransferDst, transferIn);
    commandList.CopyBufferToImage(staging, image);
    commandList.ImageBarrier(image, moe::rhi::ImageLayout::kTransferDst,
            moe::rhi::ImageLayout::kTransferSrc, transferOut);
    commandList.CopyImageToBuffer(image, readback);
    commandList.End();
    if (!device.Submit(commandList, true)) {
        return moe::test::Fail(kTestName);
    }

    {
        auto* data = static_cast<uint32_t*>(readback.Map());
        if (!data) {
            return moe::test::Fail(kTestName, "failed to map readback buffer");
        }
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            if (data[i] != i + 100u) {
                std::fprintf(stderr, "Upload smoke FAILED: data[%u] = %u, expected %u\n",
                        i, data[i], i + 100u);
                readback.Unmap();
                return moe::test::Fail(kTestName);
            }
        }
        readback.Unmap();
    }

    std::printf("Upload smoke passed.\n");

    return EXIT_SUCCESS;
}
