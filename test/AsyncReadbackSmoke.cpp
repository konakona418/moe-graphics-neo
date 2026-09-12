#include <Core/Defer.hpp>
#include <Core/Error.hpp>
#include <Core/Scheduler.hpp>
#include <Core/Task.hpp>

#include "TestSupport.hpp"

#include <Neo/AsyncReadback.hpp>
#include <Neo/TransferContext.hpp>

#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Shader.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main() {
    constexpr const char* kTestName = "Async readback smoke";
    constexpr uint32_t kElementCount = 64;
    constexpr uint64_t kByteCount = sizeof(uint32_t) * kElementCount;

    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;

    moe::rhi::Shader shader;
    moe::rhi::ShaderProgram program;
    moe::rhi::Buffer storage;
    moe::rhi::ComputePipelineState pipelineState{};
    moe::rhi::ComputePipeline pipeline;
    moe::rhi::DescriptorSetLayout layout;
    moe::rhi::DescriptorSet descriptorSet;
    moe::rhi::CommandList commandList;

    moe::Scheduler scheduler;
    moe::neo::TransferContext transfer;
    moe::neo::AsyncReadback readback;

    moe::Defer cleanup([&] {
        readback.Shutdown();
        transfer.Shutdown();
        scheduler.Shutdown();
        commandList.Destroy();
        descriptorSet.Destroy();
        storage.Destroy();
        cache.Destroy();
        device.Destroy();
    });

    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }
    if (!shader.Load(MOE_SOURCE_DIR "/shaders/rhi/sample.comp.spv",
                moe::rhi::ShaderStage::kCompute)) {
        return moe::test::Fail(kTestName);
    }
    if (!program.AddShader(shader)) {
        return moe::test::Fail(kTestName, "failed to add shader");
    }

    moe::rhi::BufferCreateInfo storageInfo{};
    storageInfo.mSize = kByteCount;
    storageInfo.mUsage = moe::rhi::BufferUsage::kStorage | moe::rhi::BufferUsage::kTransferSrc;
    if (!device.CreateBuffer(storageInfo, storage)) {
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

    if (!scheduler.Init(2)) {
        return moe::test::Fail(kTestName, "scheduler init failed");
    }
    if (!transfer.Init(device)) {
        return moe::test::Fail(kTestName);
    }
    if (!readback.Init(scheduler, transfer)) {
        return moe::test::Fail(kTestName);
    }

    // Fill the storage buffer with compute (non-blocking submit).
    commandList.Begin();
    commandList.BindDescriptorSet(pipeline, descriptorSet, 0);
    commandList.Dispatch(pipeline, kElementCount / 64, 1, 1);
    commandList.End();
    if (!device.Submit(commandList, false)) {
        return moe::test::Fail(kTestName, "compute submit failed");
    }

    // ---- poll path ----
    const moe::neo::ReadbackHandle handle = readback.Request(storage, 0, kByteCount);
    if (!handle.IsValid()) {
        return moe::test::Fail(kTestName, "request failed");
    }
    moe::neo::ReadbackLease lease;
    if (readback.TryConsume(handle, lease)) {
        return moe::test::Fail(kTestName, "readback was ready before it was submitted");
    }

    bool consumed = false;
    for (int frame = 0; frame < 3000 && !consumed; ++frame) {
        transfer.Pump();
        scheduler.Pump();
        consumed = readback.TryConsume(handle, lease);
        if (!consumed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    if (!consumed) {
        return moe::test::Fail(kTestName, "readback never completed");
    }
    {
        const auto* data = reinterpret_cast<const uint32_t*>(lease.Bytes().data());
        for (uint32_t i = 0; i < kElementCount; ++i) {
            if (data[i] != i * 2u + 1u) {
                char message[128];
                std::snprintf(message, sizeof(message), "poll data[%u] = %u, expected %u", i,
                        data[i], i * 2u + 1u);
                return moe::test::Fail(kTestName, message);
            }
        }
    }
    lease.Reset();

    // ---- coroutine path ----
    {
        moe::Task<moe::neo::ReadbackLease> task = readback.Read(storage, 0, kByteCount);
        task.Start();
        for (int frame = 0; frame < 3000 && !task.IsDone(); ++frame) {
            transfer.Pump();
            scheduler.Pump();
            if (!task.IsDone()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (!task.IsDone()) {
            return moe::test::Fail(kTestName, "coroutine readback never completed");
        }
        moe::neo::ReadbackLease coroutineLease = task.Result();
        const auto* data = reinterpret_cast<const uint32_t*>(coroutineLease.Bytes().data());
        for (uint32_t i = 0; i < kElementCount; ++i) {
            if (data[i] != i * 2u + 1u) {
                char message[128];
                std::snprintf(message, sizeof(message), "coro data[%u] = %u, expected %u", i,
                        data[i], i * 2u + 1u);
                return moe::test::Fail(kTestName, message);
            }
        }
    }

    std::printf("Async readback smoke passed.\n");
    return EXIT_SUCCESS;
}
