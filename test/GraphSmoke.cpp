#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/RenderGraph.hpp>
#include <RHI/Shader.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
    struct ImageWritePass : moe::rhi::Pass {
        moe::rhi::ComputePipeline* mPipeline{nullptr};
        moe::rhi::DescriptorSet* mSet{nullptr};

        bool Execute(moe::rhi::CommandList& cmd) override {
            cmd.BindDescriptorSet(*mPipeline, *mSet, 0);
            cmd.Dispatch(*mPipeline, 8, 8, 1);
            return true;
        }
    };

    struct ImageReadPass : moe::rhi::Pass {
        moe::rhi::Image* mImage{nullptr};
        moe::rhi::Buffer* mReadback{nullptr};

        bool Execute(moe::rhi::CommandList& cmd) override {
            cmd.CopyImageToBuffer(*mImage, *mReadback);
            return true;
        }
    };
}// namespace

int main() {
    std::string error;

    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true; // validation layer installed; must stay silent

    moe::rhi::Shader computeShader;
    moe::rhi::ShaderProgram computeProgram;
    moe::rhi::ComputePipelineState computeState{};
    moe::rhi::ComputePipeline pipeline;
    moe::rhi::DescriptorSetLayout layout;
    moe::rhi::DescriptorSet descriptorSet;

    constexpr uint32_t kSize = 64;
    moe::rhi::ImageCreateInfo imageInfo{};
    moe::rhi::Image image;
    moe::rhi::BufferCreateInfo readbackInfo{};
    moe::rhi::Buffer readback;
    moe::rhi::CommandList commandList;

    ImageWritePass writePass;
    ImageReadPass readPass;
    moe::rhi::RenderGraph graph;

    const auto imageId = graph.RegisterImage(image);
    const auto readbackId = graph.RegisterBuffer(readback);

    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        error = device.GetLastError();
        goto cleanup;
    }

    if (!computeShader.Load(MOE_SOURCE_DIR "/shaders/rhi/image_write.comp.spv", moe::rhi::ShaderStage::kCompute)) {
        error = computeShader.GetLastError();
        goto cleanup;
    }
    if (!computeProgram.AddShader(computeShader)) {
        error = "compute program add failed";
        goto cleanup;
    }
    computeState.mProgram = &computeProgram;
    if (!device.GetOrCreateComputePipeline(computeState, pipeline)) {
        error = device.GetLastError();
        goto cleanup;
    }

    // storage image written by the compute pass, then copied out
    imageInfo.mType = moe::rhi::ImageType::k2D;
    imageInfo.mWidth = kSize;
    imageInfo.mHeight = kSize;
    imageInfo.mDepth = 1;
    imageInfo.mFormat = moe::rhi::Format::kR32Uint;
    imageInfo.mUsage = moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kTransferSrc;
    if (!device.CreateImage(imageInfo, image)) {
        error = device.GetLastError();
        goto cleanup;
    }

    readbackInfo.mSize = sizeof(uint32_t) * kSize * kSize;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    if (!device.CreateBuffer(readbackInfo, readback)) {
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
    if (!descriptorSet.WriteImage(0, image, moe::rhi::DescriptorType::kStorageImage)) {
        error = "failed to write image descriptor";
        goto cleanup;
    }

    writePass.mPipeline = &pipeline;
    writePass.mSet = &descriptorSet;
    readPass.mImage = &image;
    readPass.mReadback = &readback;

    // Add the read pass FIRST to verify the graph reorders topologically.
    {
        moe::rhi::PassDesc desc{};
        desc.mName = "copy-to-buffer";
        desc.mPass = &readPass;
        moe::rhi::ResourceAccess access{};
        access.mResource = imageId;
        access.mIsWrite = false;
        access.mStage = moe::rhi::PipelineStage::kTransfer;
        access.mAccess = moe::rhi::Access::kTransferRead;
        access.mLayout = moe::rhi::ImageLayout::kTransferSrc;
        desc.mReads.push_back(access);
        access.mResource = readbackId;
        access.mIsWrite = true;
        access.mStage = moe::rhi::PipelineStage::kTransfer;
        access.mAccess = moe::rhi::Access::kTransferWrite;
        access.mLayout = moe::rhi::ImageLayout::kUndefined;
        desc.mWrites.push_back(access);
        if (!graph.AddPass(desc)) {
            error = "failed to add read pass";
            goto cleanup;
        }
    }
    {
        moe::rhi::PassDesc desc{};
        desc.mName = "compute-write";
        desc.mPass = &writePass;
        moe::rhi::ResourceAccess access{};
        access.mResource = imageId;
        access.mIsWrite = true;
        access.mStage = moe::rhi::PipelineStage::kComputeShader;
        access.mAccess = moe::rhi::Access::kShaderWrite;
        access.mLayout = moe::rhi::ImageLayout::kGeneral;
        desc.mWrites.push_back(access);
        if (!graph.AddPass(desc)) {
            error = "failed to add write pass";
            goto cleanup;
        }
    }

    if (!graph.Compile(error)) {
        goto cleanup;
    }

    if (!device.CreateCommandList(commandList)) {
        error = device.GetLastError();
        goto cleanup;
    }
    commandList.Begin();
    if (!graph.Execute(commandList)) {
        error = "graph execute failed";
        goto cleanup;
    }
    commandList.End();
    if (!device.Submit(commandList, true)) {
        error = device.GetLastError();
        goto cleanup;
    }

    {
        auto* data = static_cast<uint32_t*>(readback.Map());
        if (!data) {
            error = "failed to map readback buffer";
            goto cleanup;
        }
        // shader writes value = pixel.x + pixel.y * kSize; row-major copy makes
        // buffer[i] == i
        for (uint32_t i = 0; i < kSize * kSize; ++i) {
            if (data[i] != i) {
                std::fprintf(stderr, "Graph smoke FAILED: data[%u] = %u, expected %u\n", i, data[i], i);
                readback.Unmap();
                goto cleanup;
            }
        }
        readback.Unmap();
    }

    std::printf("Graph smoke passed.\n");

cleanup:
    commandList.Destroy();
    descriptorSet.Destroy();
    readback.Destroy();
    image.Destroy();
    cache.Destroy();
    device.Destroy();

    if (!error.empty()) {
        std::fprintf(stderr, "Graph smoke FAILED: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}