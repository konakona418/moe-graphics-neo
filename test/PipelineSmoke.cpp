#include <RHI/Device.hpp>

#include <Core/Defer.hpp>
#include "TestSupport.hpp"
#include <Core/Error.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Shader.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    constexpr const char* kTestName = "Pipeline smoke";
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};

    moe::rhi::Shader computeShader;
    moe::rhi::ShaderProgram computeProgram;
    moe::rhi::ComputePipelineState computeState{};
    moe::rhi::ComputePipeline pipelineA;
    moe::rhi::ComputePipeline pipelineB;
    moe::rhi::ComputePipeline pipelineC;

    uint32_t wx, wy, wz;

    moe::rhi::Shader vertShader;
    moe::rhi::Shader fragShader;
    moe::rhi::ShaderProgram graphicsProgram;
    moe::rhi::GraphicsPipelineState graphicsState{};
    moe::rhi::GraphicsPipeline graphicsPipeline;

    deviceInfo.mPipelineCache = &cache;

    moe::Defer cleanup([&] {
        cache.Destroy();
        device.Destroy();
    });
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }

    if (!computeShader.Load(MOE_SOURCE_DIR "/shaders/rhi/sample.comp.spv", moe::rhi::ShaderStage::kCompute)) {
        return moe::test::Fail(kTestName);
    }
    if (!computeProgram.AddShader(computeShader)) {
        return moe::test::Fail(kTestName, "compute program add failed");
    }

    computeState.mProgram = &computeProgram;
    if (!device.GetOrCreateComputePipeline(computeState, pipelineA)) {
        return moe::test::Fail(kTestName);
    }

    // cache dedup: same state must reuse the same node
    if (!device.GetOrCreateComputePipeline(computeState, pipelineB)) {
        return moe::test::Fail(kTestName);
    }
    if (cache.GetNodeCount() != 1) {
        return moe::test::Fail(kTestName, "cache dedup failed: expected 1 node");
    }

    // workgroup size reflected from the shader
    pipelineA.GetWorkgroupSize(wx, wy, wz);
    if (wx != 64 || wy != 1 || wz != 1) {
        return moe::test::Fail(kTestName, "workgroup size reflection wrong");
    }

    // graphics pipeline (fullscreen triangle, no vertex buffer)
    if (!vertShader.Load(MOE_SOURCE_DIR "/shaders/rhi/fullscreen.vert.spv", moe::rhi::ShaderStage::kVertex)) {
        return moe::test::Fail(kTestName);
    }
    if (!fragShader.Load(MOE_SOURCE_DIR "/shaders/rhi/flat.frag.spv", moe::rhi::ShaderStage::kFragment)) {
        return moe::test::Fail(kTestName);
    }
    if (!graphicsProgram.AddShader(vertShader) || !graphicsProgram.AddShader(fragShader)) {
        return moe::test::Fail(kTestName, "graphics program add failed");
    }

    graphicsState.mProgram = &graphicsProgram;
    graphicsState.mColorFormatCount = 1;
    graphicsState.mColorFormats[0] = moe::rhi::Format::kR8G8B8A8Unorm;
    graphicsState.mBlendAttachmentCount = 1;
    graphicsState.mDepth.mTestEnable = true;
    graphicsState.mDepth.mWriteEnable = true;
    graphicsState.mDepthFormat = moe::rhi::Format::kD32Float;

    if (!device.GetOrCreateGraphicsPipeline(graphicsState, graphicsPipeline)) {
        return moe::test::Fail(kTestName);
    }
    if (cache.GetNodeCount() != 2) {
        return moe::test::Fail(kTestName, "expected 2 nodes after graphics pipeline");
    }

    // manual reload invalidates the compute pipeline (and keeps the program usable)
    if (!cache.Reload(computeProgram)) {
        return moe::test::Fail(kTestName, "reload failed");
    }
    if (!device.GetOrCreateComputePipeline(computeState, pipelineC)) {
        return moe::test::Fail(kTestName);
    }

    std::printf("Pipeline smoke passed.\n");

    return EXIT_SUCCESS;
}
