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
    // All declarations at the top so every goto to cleanup below crosses no
    // non-trivial initialization.
    std::string error;

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
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        error = device.GetLastError();
        goto cleanup;
    }

    if (!shader.Load(MOE_SOURCE_DIR "/shaders/rhi/sample.comp.spv", moe::rhi::ShaderStage::kCompute)) {
        error = shader.GetLastError();
        goto cleanup;
    }
    if (!program.AddShader(shader)) {
        error = "failed to add shader to program";
        goto cleanup;
    }

    storageInfo.mSize = sizeof(uint32_t) * kElementCount;
    storageInfo.mUsage = moe::rhi::BufferUsage::kStorage | moe::rhi::BufferUsage::kTransferSrc;
    if (!device.CreateBuffer(storageInfo, storage)) {
        error = device.GetLastError();
        goto cleanup;
    }

    readbackInfo.mSize = sizeof(uint32_t) * kElementCount;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    if (!device.CreateBuffer(readbackInfo, readback)) {
        error = device.GetLastError();
        goto cleanup;
    }

    pipelineState.mProgram = &program;
    if (!device.GetOrCreateComputePipeline(pipelineState, pipeline)) {
        error = device.GetLastError();
        goto cleanup;
    }

    if (!pipeline.GetDescriptorSetLayout(0, layout)) {
        error = "no descriptor set layout 0";
        goto cleanup;
    }
    if (!device.CreateDescriptorSet(layout, descriptorSet)) {
        error = device.GetLastError();
        goto cleanup;
    }
    if (!descriptorSet.WriteBuffer(0, storage)) {
        error = "failed to write descriptor";
        goto cleanup;
    }

    if (!device.CreateCommandList(commandList)) {
        error = device.GetLastError();
        goto cleanup;
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
            error = device.GetLastError();
            goto cleanup;
        }

        auto* data = static_cast<uint32_t*>(readback.Map());
        if (!data) {
            error = "failed to map readback buffer";
            goto cleanup;
        }
        for (uint32_t i = 0; i < kElementCount; ++i) {
            if (data[i] != i * 2u + 1u) {
                std::fprintf(stderr, "Compute smoke FAILED (iter %u): data[%u] = %u, expected %u\n",
                        iteration, i, data[i], i * 2u + 1u);
                readback.Unmap();
                goto cleanup;
            }
        }
        readback.Unmap();
    }

    std::printf("Compute smoke passed.\n");

cleanup:
    commandList.Destroy();
    descriptorSet.Destroy();
    readback.Destroy();
    storage.Destroy();
    cache.Destroy();
    device.Destroy();

    if (!error.empty()) {
        std::fprintf(stderr, "Compute smoke FAILED: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}