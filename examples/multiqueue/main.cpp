#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/Queue.hpp>
#include <RHI/Shader.hpp>
#include <RHI/TimelineSemaphore.hpp>

#include <imgui.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>

namespace {
    constexpr uint32_t kElementCount = 1024;

    struct MultiQueueData {
        // compute side
        moe::rhi::Shader mComputeShader;
        moe::rhi::ShaderProgram mComputeProgram;
        moe::rhi::ComputePipeline mComputePipeline;
        moe::rhi::DescriptorSetLayout mComputeLayout;
        moe::rhi::DescriptorSet mComputeSet;
        moe::rhi::Buffer mPattern;
        moe::rhi::CommandList mComputeCmd;
        moe::rhi::TimelineSemaphore mTimeline;
        uint64_t mValue{0};

        // graphics side
        moe::rhi::Shader mVert;
        moe::rhi::Shader mFrag;
        moe::rhi::ShaderProgram mDisplayProgram;
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;

        float mTime{0.0f};
        uint32_t mGraphicsFamily{0};
        uint32_t mComputeFamily{0};
    };

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<MultiQueueData*>(userdata);
        data->mComputeFamily = ctx.mComputeQueue.GetFamily();
        {
            moe::rhi::Queue graphicsQueue;
            ctx.mDevice.GetQueue(moe::rhi::QueueType::kGraphics, graphicsQueue);
            data->mGraphicsFamily = graphicsQueue.GetFamily();
        }

        // ---- compute: waveform into a shared storage buffer ----
        if (!data->mComputeShader.Load(MOE_SOURCE_DIR "/shaders/examples/multiqueue/pattern.comp.spv",
                    moe::rhi::ShaderStage::kCompute)
                || !data->mComputeProgram.AddShader(data->mComputeShader)) {
            std::fprintf(stderr, "multiqueue: compute shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::ComputePipelineState computeState{};
        computeState.mProgram = &data->mComputeProgram;
        if (!ctx.mDevice.GetOrCreateComputePipeline(computeState, data->mComputePipeline)
                || !data->mComputePipeline.GetDescriptorSetLayout(0, data->mComputeLayout)
                || !ctx.mDevice.CreateDescriptorSet(data->mComputeLayout, data->mComputeSet)) {
            std::fprintf(stderr, "multiqueue: compute pipeline: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::BufferCreateInfo bufferInfo{};
        bufferInfo.mSize = sizeof(float) * kElementCount;
        bufferInfo.mUsage = moe::rhi::BufferUsage::kStorage;
        bufferInfo.mSharedAcrossQueues = true; // written on compute, read on graphics
        if (!ctx.mDevice.CreateBuffer(bufferInfo, data->mPattern)
                || !data->mComputeSet.WriteBuffer(0, data->mPattern)) {
            std::fprintf(stderr, "multiqueue: pattern buffer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!ctx.mDevice.CreateCommandList(moe::rhi::QueueType::kCompute, data->mComputeCmd)
                || !ctx.mDevice.CreateTimelineSemaphore(data->mTimeline)) {
            return false;
        }

        // ---- graphics: fullscreen display of the buffer ----
        if (!data->mVert.Load(MOE_SOURCE_DIR "/shaders/examples/multiqueue/display.vert.spv",
                    moe::rhi::ShaderStage::kVertex)
                || !data->mFrag.Load(MOE_SOURCE_DIR "/shaders/examples/multiqueue/display.frag.spv",
                        moe::rhi::ShaderStage::kFragment)
                || !data->mDisplayProgram.AddShader(data->mVert)
                || !data->mDisplayProgram.AddShader(data->mFrag)) {
            std::fprintf(stderr, "multiqueue: display shaders: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache, ctx.mSwapchain.GetWidth(),
                    ctx.mSwapchain.GetHeight(), ctx.mSampleCount, ctx.mTransfer)) {
            std::fprintf(stderr, "multiqueue: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<MultiQueueData*>(userdata);
        data->mTime += 0.016f;

        // Submit the compute on the compute queue, signaling a timeline value.
        data->mComputeCmd.Begin();
        data->mComputeCmd.BindDescriptorSet(data->mComputePipeline, data->mComputeSet, 0);
        data->mComputeCmd.SetPushConstants(data->mComputePipeline, 0, sizeof(float), &data->mTime);
        data->mComputeCmd.Dispatch(data->mComputePipeline, kElementCount / 64, 1, 1);
        data->mComputeCmd.End();

        data->mValue++;
        moe::rhi::TimelineSignal signal{&data->mTimeline, data->mValue};
        moe::rhi::SubmitInfo submit{};
        submit.mSignals = std::span<const moe::rhi::TimelineSignal>(&signal, 1);
        if (!ctx.mComputeQueue.Submit(data->mComputeCmd, submit)) {
            return;
        }
        // Make the graphics frame wait for that value before it samples the buffer.
        ctx.mFrameWaits.push_back(
                {&data->mTimeline, data->mValue, moe::rhi::PipelineStage::kFragmentShader});

        const float clear[4] = {0.05f, 0.05f, 0.08f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);
        const moe::neo::PassDesc displayPass{"multiqueue display", {}, {}};
        data->mRenderer.Execute(displayPass, [&](moe::neo::PassContext& pass) {
            pass.BindBuffer(0, data->mPattern);
            pass.DrawFullscreen(data->mDisplayProgram);
        });
        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<MultiQueueData*>(userdata);
        ImGui::Begin("multiqueue");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("graphics queue family: %u", data->mGraphicsFamily);
        ImGui::Text("compute queue family:  %u", data->mComputeFamily);
        ImGui::Text("compute submissions: %llu", static_cast<unsigned long long>(data->mValue));
        ImGui::TextUnformatted("compute runs on its own queue; the graphics");
        ImGui::TextUnformatted("frame waits on its timeline value.");
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<MultiQueueData*>(userdata);
        data->mRenderer.Destroy();
        data->mTimeline.Destroy();
        data->mComputeCmd.Destroy();
        data->mComputeSet.Destroy();
        data->mPattern.Destroy();
    }
}// namespace

int main() {
    MultiQueueData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("multiqueue demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "multiqueue: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
