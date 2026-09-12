#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/TransferManager.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/Shader.hpp>
#include <RHI/TimelineSemaphore.hpp>

#include <imgui.h>

#include <chrono>
#include <cstdint>
#include <span>
#include <cstdio>
#include <cstdlib>

namespace {
    // 4 MiB of results, so the GPU->CPU transfer is measurable.
    constexpr uint32_t kElementCount = 1u << 20;
    constexpr uint64_t kByteCount = sizeof(uint32_t) * kElementCount;

    struct AsyncReadbackData {
        moe::rhi::Shader mShader;
        moe::rhi::ShaderProgram mProgram;
        moe::rhi::ComputePipeline mPipeline;
        moe::rhi::DescriptorSetLayout mLayout;
        moe::rhi::DescriptorSet mSet;
        moe::rhi::Buffer mStorage;
        moe::rhi::Buffer mSyncReadback;
        moe::rhi::CommandList mCmd;
        moe::rhi::TimelineSemaphore mComputeTimeline;
        uint64_t mComputeValue{0};

        bool mSync{false};
        uint32_t mFrame{0};

        bool mHasPending{false};
        uint32_t mRequestFrame{0};
        moe::neo::ReadbackHandle mPending;

        uint64_t mChecksum{0};
        uint32_t mLatency{0};
        double mReadbackMs{0.0};
    };

    uint64_t Checksum(const uint32_t* data, uint32_t count) {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < count; ++i) {
            sum += data[i];
        }
        return sum;
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<AsyncReadbackData*>(userdata);

        if (!data->mShader.Load(MOE_SOURCE_DIR "/shaders/rhi/async_readback.comp.spv",
                    moe::rhi::ShaderStage::kCompute)) {
            std::fprintf(stderr, "async_readback: shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mProgram.AddShader(data->mShader)) {
            std::fprintf(stderr, "async_readback: add shader\n");
            return false;
        }
        moe::rhi::ComputePipelineState pipelineState{};
        pipelineState.mProgram = &data->mProgram;
        if (!ctx.mDevice.GetOrCreateComputePipeline(pipelineState, data->mPipeline)
                || !data->mPipeline.GetDescriptorSetLayout(0, data->mLayout)
                || !ctx.mDevice.CreateDescriptorSet(data->mLayout, data->mSet)) {
            std::fprintf(stderr, "async_readback: pipeline: %s\n", moe::Error::Get().c_str());
            return false;
        }

        moe::rhi::BufferCreateInfo storageInfo{};
        storageInfo.mSize = kByteCount;
        storageInfo.mUsage = moe::rhi::BufferUsage::kStorage | moe::rhi::BufferUsage::kTransferSrc;
        if (!ctx.mDevice.CreateBuffer(storageInfo, data->mStorage)
                || !data->mSet.WriteBuffer(0, data->mStorage)) {
            std::fprintf(stderr, "async_readback: storage: %s\n", moe::Error::Get().c_str());
            return false;
        }

        moe::rhi::BufferCreateInfo syncInfo{};
        syncInfo.mSize = kByteCount;
        syncInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
        syncInfo.mCpuVisible = true;
        if (!ctx.mDevice.CreateBuffer(syncInfo, data->mSyncReadback)) {
            std::fprintf(stderr, "async_readback: sync buffer: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!ctx.mDevice.CreateCommandList(moe::rhi::QueueType::kCompute, data->mCmd)
                || !ctx.mDevice.CreateTimelineSemaphore(data->mComputeTimeline)) {
            return false;
        }
        return true;
    }

    void RecordCompute(AsyncReadbackData& data) {
        // The compute queue is separate from the frame's graphics queue, so the
        // frame fence does not cover it: wait for the previous submission before
        // resetting the command buffer.
        if (data.mComputeValue > 0) {
            data.mComputeTimeline.Wait(data.mComputeValue);
        }
        data.mCmd.Begin();
        data.mCmd.BindDescriptorSet(data.mPipeline, data.mSet, 0);
        data.mCmd.SetPushConstants(data.mPipeline, 0, sizeof(uint32_t), &data.mFrame);
        data.mCmd.Dispatch(data.mPipeline, kElementCount / 64, 1, 1);
        data.mCmd.End();
        data.mComputeValue++;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList&) {
        auto* data = static_cast<AsyncReadbackData*>(userdata);
        data->mFrame++;
        RecordCompute(*data);

        const auto start = std::chrono::steady_clock::now();
        // Signal this submission's timeline value in both modes so the next
        // frame can safely reset the command buffer (and validation can see it
        // complete).
        moe::rhi::TimelineSignal signal{&data->mComputeTimeline, data->mComputeValue};
        moe::rhi::SubmitInfo submit{};
        submit.mSignals = std::span<const moe::rhi::TimelineSignal>(&signal, 1);

        if (data->mSync) {
            // Blocking path: run, copy into a CPU-visible buffer, wait, map.
            if (!ctx.mComputeQueue.Submit(data->mCmd, submit)) {
                return;
            }
            moe::rhi::CommandList copyCmd;
            if (!ctx.mDevice.CreateCommandList(moe::rhi::QueueType::kCompute, copyCmd)) {
                return;
            }
            copyCmd.Begin();
            moe::rhi::SyncInfo sync{};
            sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
            sync.mDstStage = moe::rhi::PipelineStage::kTransfer;
            sync.mDstAccess = moe::rhi::Access::kTransferRead;
            copyCmd.BufferBarrier(data->mStorage, sync);
            copyCmd.CopyBuffer(data->mStorage, data->mSyncReadback, kByteCount);
            copyCmd.End();
            // The copy is on the same queue and after the compute, so waiting
            // for it also waits for the compute.
            ctx.mComputeQueue.Submit(copyCmd, true);
            copyCmd.Destroy();

            auto* mapped = static_cast<uint32_t*>(data->mSyncReadback.Map());
            if (mapped != nullptr) {
                data->mChecksum = Checksum(mapped, kElementCount);
                data->mSyncReadback.Unmap();
            }
            data->mLatency = 0;
        } else {
            // Async path: submit and enqueue a readback; never waits.
            if (!ctx.mComputeQueue.Submit(data->mCmd, submit)) {
                return;
            }
            if (!data->mHasPending) {
                data->mPending = ctx.mTransfer.Request(data->mStorage, 0, kByteCount);
                data->mHasPending = data->mPending.IsValid();
                data->mRequestFrame = data->mFrame;
            }
        }
        data->mReadbackMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
    }

    void DrawUI(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<AsyncReadbackData*>(userdata);

        if (data->mHasPending) {
            moe::neo::ReadbackLease lease;
            if (ctx.mTransfer.TryConsume(data->mPending, lease)) {
                const auto* values = reinterpret_cast<const uint32_t*>(lease.Bytes().data());
                data->mChecksum = Checksum(values, kElementCount);
                data->mLatency = data->mFrame - data->mRequestFrame;
                data->mHasPending = false;
            }
        }

        ImGui::Begin("async readback");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Checkbox("sync readback (blocks the frame)", &data->mSync);
        ImGui::Text("readback cost: %.3f ms", data->mReadbackMs);
        ImGui::Text("checksum: %llu", static_cast<unsigned long long>(data->mChecksum));
        ImGui::Text("latency: %u frame(s)", data->mLatency);
        ImGui::Text("elements: %u (%.1f MiB)", kElementCount,
                static_cast<double>(kByteCount) / (1024.0 * 1024.0));
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<AsyncReadbackData*>(userdata);
        data->mCmd.Destroy();
        data->mComputeTimeline.Destroy();
        data->mSet.Destroy();
        data->mSyncReadback.Destroy();
        data->mStorage.Destroy();
    }
}// namespace

int main() {
    AsyncReadbackData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("async readback demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "async_readback: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
