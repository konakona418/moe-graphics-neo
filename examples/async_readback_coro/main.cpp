#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Core/Scheduler.hpp>
#include <Core/Task.hpp>
#include <Neo/TransferManager.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/Shader.hpp>
#include <RHI/TimelineSemaphore.hpp>

#include <imgui.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <thread>

namespace {
    // 4 MiB of results, so the GPU->CPU transfer is measurable.
    constexpr uint32_t kElementCount = 1u << 20;
    constexpr uint64_t kByteCount = sizeof(uint32_t) * kElementCount;

    struct CoroReadbackData {
        moe::rhi::Shader mShader;
        moe::rhi::ShaderProgram mProgram;
        moe::rhi::ComputePipeline mPipeline;
        moe::rhi::DescriptorSetLayout mLayout;
        moe::rhi::DescriptorSet mSet;
        moe::rhi::Buffer mStorage;
        moe::rhi::CommandList mCmd;
        moe::rhi::TimelineSemaphore mComputeTimeline;
        uint64_t mComputeValue{0};
        uint32_t mFrame{0};

        // The readback coroutine: co_await one readback, consume it, repeat.
        std::atomic<bool> mRunning{false};
        std::atomic<uint64_t> mCompleted{0};
        uint64_t mChecksum{0};
        moe::Task<int> mTask;
    };

    uint64_t Checksum(const uint32_t* data, uint32_t count) {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < count; ++i) {
            sum += data[i];
        }
        return sum;
    }

    // A single coroutine drives the whole readback loop: each co_await suspends
    // until the GPU copy completes, then the coroutine resumes on the main
    // thread (via the scheduler) and consumes the zero-copy lease.
    moe::Task<int> ReadbackLoop(CoroReadbackData& data, examples::AppContext& ctx) {
        uint64_t completed = 0;
        while (data.mRunning.load()) {
            moe::neo::ReadbackLease lease =
                    co_await ctx.mTransfer.Read(data.mStorage, 0, kByteCount);
            if (lease) {
                const auto* values = reinterpret_cast<const uint32_t*>(lease.Bytes().data());
                data.mChecksum = Checksum(values, kElementCount);
                data.mCompleted.store(++completed);
            }
        }
        co_return static_cast<int>(completed);
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<CoroReadbackData*>(userdata);

        if (!data->mShader.Load(MOE_SOURCE_DIR "/shaders/rhi/async_readback.comp.spv",
                    moe::rhi::ShaderStage::kCompute)
                || !data->mProgram.AddShader(data->mShader)) {
            std::fprintf(stderr, "async_readback_coro: shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::ComputePipelineState pipelineState{};
        pipelineState.mProgram = &data->mProgram;
        if (!ctx.mDevice.GetOrCreateComputePipeline(pipelineState, data->mPipeline)
                || !data->mPipeline.GetDescriptorSetLayout(0, data->mLayout)
                || !ctx.mDevice.CreateDescriptorSet(data->mLayout, data->mSet)) {
            std::fprintf(stderr, "async_readback_coro: pipeline: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::BufferCreateInfo storageInfo{};
        storageInfo.mSize = kByteCount;
        storageInfo.mUsage = moe::rhi::BufferUsage::kStorage | moe::rhi::BufferUsage::kTransferSrc;
        if (!ctx.mDevice.CreateBuffer(storageInfo, data->mStorage)
                || !data->mSet.WriteBuffer(0, data->mStorage)) {
            std::fprintf(stderr, "async_readback_coro: storage: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!ctx.mDevice.CreateCommandList(moe::rhi::QueueType::kCompute, data->mCmd)
                || !ctx.mDevice.CreateTimelineSemaphore(data->mComputeTimeline)) {
            return false;
        }

        data->mRunning = true;
        data->mTask = ReadbackLoop(*data, ctx);
        data->mTask.Start();
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList&) {
        auto* data = static_cast<CoroReadbackData*>(userdata);
        data->mFrame++;

        // The compute queue is separate from graphics, so wait for the previous
        // submission before resetting the command buffer.
        if (data->mComputeValue > 0) {
            data->mComputeTimeline.Wait(data->mComputeValue);
        }
        data->mCmd.Begin();
        data->mCmd.BindDescriptorSet(data->mPipeline, data->mSet, 0);
        data->mCmd.SetPushConstants(data->mPipeline, 0, sizeof(uint32_t), &data->mFrame);
        data->mCmd.Dispatch(data->mPipeline, kElementCount / 64, 1, 1);
        data->mCmd.End();
        data->mComputeValue++;

        moe::rhi::TimelineSignal signal{&data->mComputeTimeline, data->mComputeValue};
        moe::rhi::SubmitInfo submit{};
        submit.mSignals = std::span<const moe::rhi::TimelineSignal>(&signal, 1);
        ctx.mComputeQueue.Submit(data->mCmd, submit);
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<CoroReadbackData*>(userdata);
        ImGui::Begin("async readback (coroutine)");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("completed readbacks: %llu",
                static_cast<unsigned long long>(data->mCompleted.load()));
        ImGui::Text("checksum: %llu", static_cast<unsigned long long>(data->mChecksum));
        ImGui::Text("elements: %u (%.1f MiB)", kElementCount,
                static_cast<double>(kByteCount) / (1024.0 * 1024.0));
        ImGui::TextUnformatted("one coroutine co_awaits each readback;");
        ImGui::TextUnformatted("it resumes on the main thread via the scheduler.");
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<CoroReadbackData*>(userdata);
        // Let the coroutine observe the stop flag and finish: pump until it is
        // done so no continuation outlives its frame.
        data->mRunning = false;
        for (int i = 0; i < 20000 && !data->mTask.IsDone(); ++i) {
            ctx.mTransfer.Pump();
            ctx.mScheduler.Pump();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        data->mCmd.Destroy();
        data->mComputeTimeline.Destroy();
        data->mSet.Destroy();
        data->mStorage.Destroy();
    }
}// namespace

int main() {
    CoroReadbackData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("async readback (coroutine)", 1280, 720, callbacks)) {
        std::fprintf(stderr, "async_readback_coro: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
