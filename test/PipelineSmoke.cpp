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
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        error = device.GetLastError();
        goto cleanup;
    }

    if (!computeShader.Load(MOE_SOURCE_DIR "/shaders/rhi/sample.comp.spv", moe::rhi::ShaderStage::kCompute)) {
        error = computeShader.GetLastError();
        goto cleanup;
    }
    if (!computeProgram.AddShader(computeShader)) {
        error = "compute program add failed";
        goto cleanup;
    }

    computeState.mProgram = &computeProgram;
    if (!device.GetOrCreateComputePipeline(computeState, pipelineA)) {
        error = device.GetLastError();
        goto cleanup;
    }

    // cache dedup: same state must reuse the same node
    if (!device.GetOrCreateComputePipeline(computeState, pipelineB)) {
        error = device.GetLastError();
        goto cleanup;
    }
    if (cache.GetNodeCount() != 1) {
        error = "cache dedup failed: expected 1 node";
        goto cleanup;
    }

    // workgroup size reflected from the shader
    pipelineA.GetWorkgroupSize(wx, wy, wz);
    if (wx != 64 || wy != 1 || wz != 1) {
        error = "workgroup size reflection wrong";
        goto cleanup;
    }

    // graphics pipeline (fullscreen triangle, no vertex buffer)
    if (!vertShader.Load(MOE_SOURCE_DIR "/shaders/rhi/fullscreen.vert.spv", moe::rhi::ShaderStage::kVertex)) {
        error = vertShader.GetLastError();
        goto cleanup;
    }
    if (!fragShader.Load(MOE_SOURCE_DIR "/shaders/rhi/flat.frag.spv", moe::rhi::ShaderStage::kFragment)) {
        error = fragShader.GetLastError();
        goto cleanup;
    }
    if (!graphicsProgram.AddShader(vertShader) || !graphicsProgram.AddShader(fragShader)) {
        error = "graphics program add failed";
        goto cleanup;
    }

    graphicsState.mProgram = &graphicsProgram;
    graphicsState.mColorFormatCount = 1;
    graphicsState.mColorFormats[0] = moe::rhi::Format::kR8G8B8A8Unorm;
    graphicsState.mBlendAttachmentCount = 1;
    graphicsState.mDepth.mTestEnable = true;
    graphicsState.mDepth.mWriteEnable = true;
    graphicsState.mDepthFormat = moe::rhi::Format::kD32Float;

    if (!device.GetOrCreateGraphicsPipeline(graphicsState, graphicsPipeline)) {
        error = device.GetLastError();
        goto cleanup;
    }
    if (cache.GetNodeCount() != 2) {
        error = "expected 2 nodes after graphics pipeline";
        goto cleanup;
    }

    // manual reload invalidates the compute pipeline (and keeps the program usable)
    if (!cache.Reload(computeProgram)) {
        error = "reload failed";
        goto cleanup;
    }
    if (!device.GetOrCreateComputePipeline(computeState, pipelineC)) {
        error = device.GetLastError();
        goto cleanup;
    }

    std::printf("Pipeline smoke passed.\n");

cleanup:
    cache.Destroy();
    device.Destroy();

    if (!error.empty()) {
        std::fprintf(stderr, "Pipeline smoke FAILED: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}