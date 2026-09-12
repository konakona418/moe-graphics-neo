#include "Neo/TransferContext.hpp"
#include <Core/Profile.hpp>

#include <Core/Error.hpp>
#include <RHI/CommandList.hpp>

#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace moe::neo {
    namespace {
        struct StagingSlot {
            rhi::Buffer mBuffer;
            uint64_t mSize{0};
            std::byte* mMapped{nullptr};
            bool mInUse{false};
        };
    }// namespace

    struct TransferContext::Impl {
        struct PendingReadback {
            const rhi::Buffer* mSrc{nullptr};
            uint64_t mOffset{0};
            uint64_t mSize{0};
            TransferSlotId mSlot{kInvalidTransferSlot};
            rhi::PipelineStage mSrcStage{rhi::PipelineStage::kComputeShader};
            rhi::Access mSrcAccess{rhi::Access::kShaderWrite};
            ReadbackCallback mOnComplete;
        };

        struct Completion {
            uint64_t mValue{0};
            TransferSlotId mSlot{kInvalidTransferSlot};
            ReadbackCallback mOnComplete;
        };

        rhi::Device* mDevice{nullptr};
        rhi::TimelineSemaphore mCompletion;
        uint64_t mNextValue{0};

        std::mutex mPoolMutex;
        std::vector<std::unique_ptr<StagingSlot>> mSlots;

        std::mutex mPendingMutex;
        std::vector<PendingReadback> mPending;

        std::mutex mCompletionMutex;
        std::condition_variable mCompletionCv;
        std::deque<Completion> mCompletions; // ascending value
        std::thread mCompletionThread;
        bool mThreadStarted{false};
        bool mStopping{false};
        bool mRunning{false};

        // The completion thread is started lazily so contexts that only do
        // synchronous uploads never pay for a thread.
        void EnsureThread() {
            if (!mThreadStarted) {
                mCompletionThread = std::thread([this] { CompletionLoop(); });
                mThreadStarted = true;
            }
        }

        TransferSlotId AcquireSlot(uint64_t size) {
            std::lock_guard<std::mutex> lock(mPoolMutex);
            for (uint32_t i = 0; i < mSlots.size(); ++i) {
                if (!mSlots[i]->mInUse && mSlots[i]->mSize >= size) {
                    mSlots[i]->mInUse = true;
                    return i;
                }
            }
            auto slot = std::make_unique<StagingSlot>();
            rhi::BufferCreateInfo info{};
            info.mSize = size;
            info.mUsage = rhi::BufferUsage::kTransferSrc | rhi::BufferUsage::kTransferDst;
            info.mCpuVisible = true;
            if (!mDevice->CreateBuffer(info, slot->mBuffer)) {
                return kInvalidTransferSlot;
            }
            slot->mSize = size;
            slot->mInUse = true;
            mSlots.push_back(std::move(slot));
            return static_cast<TransferSlotId>(mSlots.size() - 1);
        }

        std::byte* MapSlot(TransferSlotId slot) {
            std::lock_guard<std::mutex> lock(mPoolMutex);
            if (slot >= mSlots.size()) {
                return nullptr;
            }
            StagingSlot& staging = *mSlots[slot];
            staging.mMapped = static_cast<std::byte*>(staging.mBuffer.Map());
            return staging.mMapped;
        }

        void UnmapSlot(TransferSlotId slot) {
            std::lock_guard<std::mutex> lock(mPoolMutex);
            if (slot >= mSlots.size()) {
                return;
            }
            StagingSlot& staging = *mSlots[slot];
            if (staging.mMapped != nullptr) {
                staging.mBuffer.Unmap();
                staging.mMapped = nullptr;
            }
        }

        void ReleaseSlot(TransferSlotId slot) {
            std::lock_guard<std::mutex> lock(mPoolMutex);
            if (slot >= mSlots.size()) {
                return;
            }
            StagingSlot& staging = *mSlots[slot];
            if (staging.mMapped != nullptr) {
                staging.mBuffer.Unmap();
                staging.mMapped = nullptr;
            }
            staging.mInUse = false;
        }

        void RunCompletion(const Completion& completion) {
            if (completion.mOnComplete) {
                std::byte* mapped = MapSlot(completion.mSlot);
                uint64_t size = 0;
                {
                    std::lock_guard<std::mutex> lock(mPoolMutex);
                    if (completion.mSlot < mSlots.size()) {
                        size = mSlots[completion.mSlot]->mSize;
                    }
                }
                completion.mOnComplete(completion.mSlot, mapped, size);
            } else {
                ReleaseSlot(completion.mSlot);
            }
        }

        void CompletionLoop() {
            std::unique_lock<std::mutex> lock(mCompletionMutex);
            for (;;) {
                mCompletionCv.wait(lock, [this] { return mStopping || !mCompletions.empty(); });
                if (mCompletions.empty()) {
                    if (mStopping) {
                        return;
                    }
                    continue;
                }
                const uint64_t target = mCompletions.front().mValue;
                lock.unlock();
                mCompletion.Wait(target, 100'000'000ull); // 100 ms
                lock.lock();
                const uint64_t reached = mCompletion.GetValue();
                while (!mCompletions.empty() && mCompletions.front().mValue <= reached) {
                    Completion completion = std::move(mCompletions.front());
                    mCompletions.pop_front();
                    lock.unlock();
                    RunCompletion(completion);
                    lock.lock();
                }
                mCompletionCv.notify_all();
            }
        }
    };

    TransferContext::TransferContext() = default;

    TransferContext::~TransferContext() {
        Shutdown();
    }

    bool TransferContext::Init(rhi::Device& device) {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mRunning) {
            return false;
        }
        if (mImpl == nullptr) {
            mImpl = std::make_unique<Impl>();
        }
        mImpl->mDevice = &device;
        if (!device.CreateTimelineSemaphore(mImpl->mCompletion)) {
            return moe::Fail("TransferContext: timeline semaphore creation failed: "
                    + moe::Error::Get());
        }
        mImpl->mStopping = false;
        mImpl->mRunning = true;
        return true;
    }

    void TransferContext::Shutdown() {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return;
        }
        if (mImpl->mThreadStarted) {
            {
                std::lock_guard<std::mutex> lock(mImpl->mCompletionMutex);
                mImpl->mStopping = true;
            }
            mImpl->mCompletionCv.notify_all();
            if (mImpl->mCompletionThread.joinable()) {
                mImpl->mCompletionThread.join();
            }
            mImpl->mThreadStarted = false;
        }
        for (auto& slot : mImpl->mSlots) {
            slot->mBuffer.Destroy();
        }
        mImpl->mSlots.clear();
        mImpl->mCompletion.Destroy();
        mImpl->mRunning = false;
    }

    bool TransferContext::Upload(const uint8_t* data, size_t byteCount, const rhi::Buffer& dst,
            rhi::PipelineStage dstStage, rhi::Access dstAccess, bool waitForCompletion) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferContext: not initialized");
        }
        const TransferSlotId slot = mImpl->AcquireSlot(byteCount);
        if (slot == kInvalidTransferSlot) {
            return moe::Fail("TransferContext: staging allocation failed");
        }
        std::byte* mapped = mImpl->MapSlot(slot);
        if (mapped == nullptr) {
            mImpl->ReleaseSlot(slot);
            return moe::Fail("TransferContext: staging map failed");
        }
        std::memcpy(mapped, data, byteCount);
        mImpl->UnmapSlot(slot);

        rhi::CommandList cmd;
        if (!mImpl->mDevice->CreateCommandList(cmd)) {
            mImpl->ReleaseSlot(slot);
            return moe::Fail("TransferContext: command list creation failed");
        }
        cmd.Begin();
        cmd.CopyBuffer(mImpl->mSlots[slot]->mBuffer, dst, byteCount, 0, 0);
        rhi::SyncInfo sync{};
        sync.mSrcStage = rhi::PipelineStage::kTransfer;
        sync.mSrcAccess = rhi::Access::kTransferWrite;
        sync.mDstStage = dstStage;
        sync.mDstAccess = dstAccess;
        cmd.BufferBarrier(dst, sync);
        cmd.End();

        const uint64_t value = ++mImpl->mNextValue;
        rhi::TimelineSignal signal{&mImpl->mCompletion, value};
        rhi::SubmitInfo submit{};
        submit.mSignals = std::span<const rhi::TimelineSignal>(&signal, 1);
        if (!mImpl->mDevice->Submit(cmd, submit)) {
            cmd.Destroy();
            mImpl->ReleaseSlot(slot);
            return moe::Fail("TransferContext: submit failed: " + moe::Error::Get());
        }
        cmd.Destroy();

        if (waitForCompletion) {
            mImpl->mCompletion.Wait(value);
            mImpl->ReleaseSlot(slot);
        } else {
            mImpl->EnsureThread();
            std::lock_guard<std::mutex> lock(mImpl->mCompletionMutex);
            mImpl->mCompletions.push_back(Impl::Completion{value, slot, nullptr});
            mImpl->mCompletionCv.notify_one();
        }
        return true;
    }

    bool TransferContext::EnqueueReadback(const rhi::Buffer& src, uint64_t offset, uint64_t size,
            rhi::PipelineStage srcStage, rhi::Access srcAccess, ReadbackCallback onComplete) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferContext: not initialized");
        }
        const TransferSlotId slot = mImpl->AcquireSlot(size);
        if (slot == kInvalidTransferSlot) {
            return moe::Fail("TransferContext: staging allocation failed");
        }
        mImpl->EnsureThread();
        std::lock_guard<std::mutex> lock(mImpl->mPendingMutex);
        mImpl->mPending.push_back(Impl::PendingReadback{
                &src, offset, size, slot, srcStage, srcAccess, std::move(onComplete)});
        return true;
    }

    std::byte* TransferContext::MapSlot(TransferSlotId slot) {
        return mImpl != nullptr ? mImpl->MapSlot(slot) : nullptr;
    }

    void TransferContext::ReleaseSlot(TransferSlotId slot) {
        if (mImpl != nullptr) {
            mImpl->ReleaseSlot(slot);
        }
    }

    void TransferContext::Pump() {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return;
        }
        std::vector<Impl::PendingReadback> pending;
        {
            std::lock_guard<std::mutex> lock(mImpl->mPendingMutex);
            pending.swap(mImpl->mPending);
        }
        for (auto& request : pending) {
            rhi::CommandList cmd;
            if (!mImpl->mDevice->CreateCommandList(cmd)) {
                mImpl->ReleaseSlot(request.mSlot);
                continue;
            }
            cmd.Begin();
            rhi::SyncInfo sync{};
            sync.mSrcStage = request.mSrcStage;
            sync.mSrcAccess = request.mSrcAccess;
            sync.mDstStage = rhi::PipelineStage::kTransfer;
            sync.mDstAccess = rhi::Access::kTransferRead;
            cmd.BufferBarrier(*request.mSrc, sync);
            cmd.CopyBuffer(*request.mSrc, mImpl->mSlots[request.mSlot]->mBuffer, request.mSize,
                    request.mOffset, 0);
            cmd.End();

            const uint64_t value = ++mImpl->mNextValue;
            rhi::TimelineSignal signal{&mImpl->mCompletion, value};
            rhi::SubmitInfo submit{};
            submit.mSignals = std::span<const rhi::TimelineSignal>(&signal, 1);
            if (!mImpl->mDevice->Submit(cmd, submit)) {
                cmd.Destroy();
                mImpl->ReleaseSlot(request.mSlot);
                continue;
            }
            cmd.Destroy();

            std::lock_guard<std::mutex> lock(mImpl->mCompletionMutex);
            mImpl->mCompletions.push_back(
                    Impl::Completion{value, request.mSlot, std::move(request.mOnComplete)});
            mImpl->mCompletionCv.notify_one();
        }
    }

    void TransferContext::Drain() {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr) {
            return;
        }
        Pump();
        std::unique_lock<std::mutex> lock(mImpl->mCompletionMutex);
        mImpl->mCompletionCv.wait(lock, [this] { return mImpl->mCompletions.empty(); });
    }
}// namespace moe::neo
