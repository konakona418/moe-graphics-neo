#include <Core/Error.hpp>

#include <Core/Defer.hpp>
#include "TestSupport.hpp"

#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Shader.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    constexpr const char* kTestName = "Compute smoke";
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};

    moe::rhi::Shader shader;
    moe::rhi::ShaderProgram program;

    constexpr uint32_t kElementCount = 64;

    moe::rhi::BufferCreateInfo storageInfo{};
    moe::rhi::Buffer storage;
    moe::rhi::BufferCreateInfo readbackInfo{};
    moe::rhi::Buffer readback;

    moe::rhi::ComputePipelineState pipelineState{};
    moe::rhi::ComputePipeline pipeline;
    moe::rhi::DescriptorSetLayout layout;
    moe::rhi::DescriptorSet descriptorSet;
    moe::rhi::CommandList commandList;

    moe::rhi::SyncInfo computeToTransfer{};
    computeToTransfer.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
    computeToTransfer.mSrcAccess = moe::rhi::Access::kShaderWrite;
    computeToTransfer.mDstStage = moe::rhi::PipelineStage::kTransfer;
    computeToTransfer.mDstAccess = moe::rhi::Access::kTransferRead;

    deviceInfo.mPipelineCache = &cache;

    moe::Defer cleanup([&] {
        commandList.Destroy();
        descriptorSet.Destroy();
        readback.Destroy();
        storage.Destroy();
        cache.Destroy();
        device.Destroy();
    });
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }

    if (!shader.Load(MOE_SOURCE_DIR "/shaders/rhi/sample.comp.spv", moe::rhi::ShaderStage::kCompute)) {
        return moe::test::Fail(kTestName);
    }
    if (!program.AddShader(shader)) {
        return moe::test::Fail(kTestName, "failed to add shader to program");
    }

    storageInfo.mSize = sizeof(uint32_t) * kElementCount;
    storageInfo.mUsage = moe::rhi::BufferUsage::kStorage | moe::rhi::BufferUsage::kTransferSrc;
    if (!device.CreateBuffer(storageInfo, storage)) {
        return moe::test::Fail(kTestName);
    }

    readbackInfo.mSize = sizeof(uint32_t) * kElementCount;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    if (!device.CreateBuffer(readbackInfo, readback)) {
        return moe::test::Fail(kTestName);
    }

    pipelineState.mProgram = &program;
    if (!device.GetOrCreateComputePipeline(pipelineState, pipeline)) {
        return moe::test::Fail(kTestName);
    }

    if (!pipeline.GetDescriptorSetLayout(0, layout)) {
        return moe::test::Fail(kTestName, "no descriptor set layout 0");
    }
    if (!device.CreateDescriptorSet(layout, descriptorSet)) {
        return moe::test::Fail(kTestName);
    }
    if (!descriptorSet.WriteBuffer(0, storage)) {
        return moe::test::Fail(kTestName, "failed to write descriptor");
    }

    if (!device.CreateCommandList(commandList)) {
        return moe::test::Fail(kTestName);
    }

    // Several iterations shake out sync races (compute write -> buffer barrier
    // -> transfer read).
    for (uint32_t iteration = 0; iteration < 5; ++iteration) {
        commandList.Begin();
        commandList.BindDescriptorSet(pipeline, descriptorSet, 0);
        commandList.Dispatch(pipeline, kElementCount / 64, 1, 1);
        commandList.BufferBarrier(storage, computeToTransfer);
        commandList.CopyBuffer(storage, readback, storage.GetSize());
        commandList.End();
        if (!device.Submit(commandList, true)) {
            return moe::test::Fail(kTestName);
        }

        auto* data = static_cast<uint32_t*>(readback.Map());
        if (!data) {
            return moe::test::Fail(kTestName, "failed to map readback buffer");
        }
        for (uint32_t i = 0; i < kElementCount; ++i) {
            if (data[i] != i * 2u + 1u) {
                char message[128];
                std::snprintf(message, sizeof(message),
                        "data[%u] = %u, expected %u (iter %u)", i, data[i], i * 2u + 1u, iteration);
                readback.Unmap();
                return moe::test::Fail(kTestName, message);
            }
        }
        readback.Unmap();
    }

    std::printf("Compute smoke passed.\n");

    return EXIT_SUCCESS;
}
