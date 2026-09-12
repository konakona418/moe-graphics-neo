#include <Core/Defer.hpp>
#include <Core/Error.hpp>
#include <Core/Scheduler.hpp>

#include "TestSupport.hpp"

#include <Neo/TransferManager.hpp>

#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Fence.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Queue.hpp>
#include <RHI/Shader.hpp>
#include <RHI/TimelineSemaphore.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main() {
    constexpr const char* kTestName = "Cross-queue smoke";
    constexpr uint32_t kElementCount = 64;
    constexpr uint64_t kByteCount = sizeof(uint32_t) * kElementCount;

    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;

    moe::rhi::Shader shader;
    moe::rhi::ShaderProgram program;
    moe::rhi::Buffer storage;
    moe::rhi::Buffer readback;
    moe::rhi::ComputePipelineState pipelineState{};
    moe::rhi::ComputePipeline pipeline;
    moe::rhi::DescriptorSetLayout layout;
    moe::rhi::DescriptorSet descriptorSet;
    moe::rhi::CommandList computeCmd;
    moe::rhi::CommandList graphicsCmd;
    moe::rhi::Queue computeQueue;
    moe::rhi::Queue graphicsQueue;
    moe::rhi::TimelineSemaphore timeline;
    moe::rhi::Fence fence;

    moe::Scheduler scheduler;
    moe::neo::TransferManager transfer;

    moe::Defer cleanup([&] {
        transfer.Shutdown();
        scheduler.Shutdown();
        fence.Destroy();
        timeline.Destroy();
        graphicsCmd.Destroy();
        computeCmd.Destroy();
        descriptorSet.Destroy();
        readback.Destroy();
        storage.Destroy();
        cache.Destroy();
        device.Destroy();
    });

    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }
    if (!device.GetQueue(moe::rhi::QueueType::kCompute, computeQueue)
            || !device.GetQueue(moe::rhi::QueueType::kGraphics, graphicsQueue)) {
        return moe::test::Fail(kTestName, "failed to get queues");
    }
    if (computeQueue.GetFamily() == graphicsQueue.GetFamily()) {
        std::printf("Cross-queue smoke SKIPPED (unified queue family).\n");
        return EXIT_SUCCESS;
    }

    if (!shader.Load(MOE_SOURCE_DIR "/shaders/rhi/sample.comp.spv",
                moe::rhi::ShaderStage::kCompute)
            || !program.AddShader(shader)) {
        return moe::test::Fail(kTestName);
    }

    // Compute writes this; graphics reads it -> concurrent sharing.
    moe::rhi::BufferCreateInfo storageInfo{};
    storageInfo.mSize = kByteCount;
    storageInfo.mUsage = moe::rhi::BufferUsage::kStorage | moe::rhi::BufferUsage::kTransferSrc;
    storageInfo.mSharedAcrossQueues = true;
    if (!device.CreateBuffer(storageInfo, storage)) {
        return moe::test::Fail(kTestName);
    }

    moe::rhi::BufferCreateInfo readbackInfo{};
    readbackInfo.mSize = kByteCount;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    if (!device.CreateBuffer(readbackInfo, readback)) {
        return moe::test::Fail(kTestName);
    }

    pipelineState.mProgram = &program;
    if (!device.GetOrCreateComputePipeline(pipelineState, pipeline)
            || !pipeline.GetDescriptorSetLayout(0, layout)
            || !device.CreateDescriptorSet(layout, descriptorSet)
            || !descriptorSet.WriteBuffer(0, storage)) {
        return moe::test::Fail(kTestName);
    }

    if (!device.CreateCommandList(moe::rhi::QueueType::kCompute, computeCmd)
            || !device.CreateCommandList(moe::rhi::QueueType::kGraphics, graphicsCmd)) {
        return moe::test::Fail(kTestName);
    }
    if (!device.CreateTimelineSemaphore(timeline) || !device.CreateFence(fence)) {
        return moe::test::Fail(kTestName);
    }

    // ---- compute on the compute queue, signal the timeline ----
    computeCmd.Begin();
    computeCmd.BindDescriptorSet(pipeline, descriptorSet, 0);
    computeCmd.Dispatch(pipeline, kElementCount / 64, 1, 1);
    computeCmd.End();
    {
        moe::rhi::TimelineSignal signal{&timeline, 1};
        moe::rhi::SubmitInfo submit{};
        submit.mSignals = std::span<const moe::rhi::TimelineSignal>(&signal, 1);
        if (!computeQueue.Submit(computeCmd, submit)) {
            return moe::test::Fail(kTestName, "compute submit failed");
        }
    }

    // ---- graphics waits on that value, then copies to a host buffer ----
    graphicsCmd.Begin();
    {
        moe::rhi::SyncInfo sync{};
        sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
        sync.mDstStage = moe::rhi::PipelineStage::kTransfer;
        sync.mDstAccess = moe::rhi::Access::kTransferRead;
        graphicsCmd.BufferBarrier(storage, sync);
    }
    graphicsCmd.CopyBuffer(storage, readback, kByteCount);
    graphicsCmd.End();
    {
        moe::rhi::TimelineWait wait{&timeline, 1, moe::rhi::PipelineStage::kTransfer};
        moe::rhi::SubmitInfo submit{};
        submit.mWaits = std::span<const moe::rhi::TimelineWait>(&wait, 1);
        if (!graphicsQueue.Submit(graphicsCmd, submit, &fence)) {
            return moe::test::Fail(kTestName, "graphics submit failed");
        }
        if (!fence.Wait()) {
            return moe::test::Fail(kTestName, "fence wait failed");
        }
    }
    {
        const auto* data = static_cast<const uint32_t*>(readback.Map());
        if (data == nullptr) {
            return moe::test::Fail(kTestName, "failed to map readback");
        }
        for (uint32_t i = 0; i < kElementCount; ++i) {
            if (data[i] != i * 2u + 1u) {
                char message[128];
                std::snprintf(message, sizeof(message),
                        "cross-queue data[%u] = %u, expected %u", i, data[i], i * 2u + 1u);
                readback.Unmap();
                return moe::test::Fail(kTestName, message);
            }
        }
        readback.Unmap();
    }

    // ---- async readback of the compute-written buffer (same family) ----
    if (!scheduler.Init(2) || !transfer.Init(device, scheduler)) {
        return moe::test::Fail(kTestName, "transfer init failed");
    }
    const moe::neo::ReadbackHandle handle = transfer.Request(storage, 0, kByteCount);
    if (!handle.IsValid()) {
        return moe::test::Fail(kTestName, "readback request failed");
    }
    bool consumed = false;
    moe::neo::ReadbackLease lease;
    for (int frame = 0; frame < 3000 && !consumed; ++frame) {
        transfer.Pump();
        scheduler.Pump();
        consumed = transfer.TryConsume(handle, lease);
        if (!consumed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    if (!consumed) {
        return moe::test::Fail(kTestName, "async readback never completed");
    }
    {
        const auto* data = reinterpret_cast<const uint32_t*>(lease.Bytes().data());
        for (uint32_t i = 0; i < kElementCount; ++i) {
            if (data[i] != i * 2u + 1u) {
                char message[128];
                std::snprintf(message, sizeof(message),
                        "readback data[%u] = %u, expected %u", i, data[i], i * 2u + 1u);
                return moe::test::Fail(kTestName, message);
            }
        }
    }

    std::printf("Cross-queue smoke passed.\n");
    return EXIT_SUCCESS;
}
