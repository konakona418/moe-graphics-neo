#include "Neo/AsyncReadback.hpp"
#include <Core/Profile.hpp>

#include <Core/AsyncEvent.hpp>
#include <Core/Error.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace moe::neo {
    namespace {
        struct ReadbackRequest {
            explicit ReadbackRequest(moe::Scheduler& scheduler) : mEvent(scheduler) {}

            moe::AsyncEvent<ReadbackLease> mEvent;
            TransferSlotId mSlot{kInvalidTransferSlot};
            uint32_t mGeneration{0};
            bool mActive{false};
        };
    }// namespace

    struct AsyncReadback::Impl {
        moe::Scheduler* mScheduler{nullptr};
        TransferContext* mTransfer{nullptr};
        std::vector<std::shared_ptr<ReadbackRequest>> mSlots;
        std::vector<uint32_t> mFree;
        bool mRunning{false};

        uint32_t Alloc() {
            if (!mFree.empty()) {
                const uint32_t index = mFree.back();
                mFree.pop_back();
                return index;
            }
            mSlots.push_back(std::make_shared<ReadbackRequest>(*mScheduler));
            return static_cast<uint32_t>(mSlots.size() - 1);
        }

        void Free(uint32_t index) {
            mSlots[index]->mActive = false;
            mSlots[index]->mGeneration++;
            mFree.push_back(index);
        }
    };

    ReadbackLease::ReadbackLease(std::span<std::byte> bytes, TransferContext* owner, TransferSlotId slot)
        : mBytes(bytes), mOwner(owner), mSlot(slot) {}

    ReadbackLease::~ReadbackLease() {
        Reset();
    }

    ReadbackLease::ReadbackLease(ReadbackLease&& other) noexcept
        : mBytes(other.mBytes), mOwner(other.mOwner), mSlot(other.mSlot) {
        other.mBytes = {};
        other.mOwner = nullptr;
        other.mSlot = kInvalidTransferSlot;
    }

    ReadbackLease& ReadbackLease::operator=(ReadbackLease&& other) noexcept {
        if (this != &other) {
            Reset();
            mBytes = other.mBytes;
            mOwner = other.mOwner;
            mSlot = other.mSlot;
            other.mBytes = {};
            other.mOwner = nullptr;
            other.mSlot = kInvalidTransferSlot;
        }
        return *this;
    }

    void ReadbackLease::Reset() {
        if (mOwner != nullptr) {
            mOwner->ReleaseSlot(mSlot);
            mOwner = nullptr;
            mBytes = {};
            mSlot = kInvalidTransferSlot;
        }
    }

    AsyncReadback::AsyncReadback() = default;

    AsyncReadback::~AsyncReadback() {
        Shutdown();
    }

    bool AsyncReadback::Init(moe::Scheduler& scheduler, TransferContext& transfer) {
        if (mImpl && mImpl->mRunning) {
            return false;
        }
        if (mImpl == nullptr) {
            mImpl = std::make_unique<Impl>();
        }
        mImpl->mScheduler = &scheduler;
        mImpl->mTransfer = &transfer;
        mImpl->mRunning = true;
        return true;
    }

    void AsyncReadback::Shutdown() {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return;
        }
        // Process every in-flight readback so no completion callback outlives
        // the requests it resolves. The GPU must be idle first.
        mImpl->mTransfer->Drain();
        mImpl->mSlots.clear();
        mImpl->mFree.clear();
        mImpl->mRunning = false;
    }

    ReadbackHandle AsyncReadback::Request(const rhi::Buffer& src, uint64_t offset, uint64_t size,
            rhi::PipelineStage srcStage, rhi::Access srcAccess) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            moe::Error::Set("AsyncReadback: not initialized");
            return {};
        }
        const uint32_t index = mImpl->Alloc();
        const std::shared_ptr<ReadbackRequest> request = mImpl->mSlots[index];
        request->mActive = true;
        request->mSlot = kInvalidTransferSlot;

        TransferContext* transfer = mImpl->mTransfer;
        const bool ok = transfer->EnqueueReadback(src, offset, size, srcStage, srcAccess,
                [request, transfer](TransferSlotId slot, std::byte* mapped, uint64_t bytes) {
                    request->mSlot = slot;
                    const bool valid = mapped != nullptr;
                    const std::span<std::byte> span = valid
                            ? std::span<std::byte>(mapped, bytes)
                            : std::span<std::byte>{};
                    request->mEvent.SetValue(
                            ReadbackLease{span, valid ? transfer : nullptr, slot});
                });
        if (!ok) {
            mImpl->Free(index);
            return {};
        }
        return ReadbackHandle{index, request->mGeneration};
    }

    bool AsyncReadback::TryConsume(ReadbackHandle handle, ReadbackLease& out) {
        if (mImpl == nullptr || !mImpl->mRunning || handle.mIndex >= mImpl->mSlots.size()) {
            return false;
        }
        const std::shared_ptr<ReadbackRequest> request = mImpl->mSlots[handle.mIndex];
        if (!request->mActive || request->mGeneration != handle.mGeneration) {
            return false;
        }
        std::optional<ReadbackLease> value = request->mEvent.TryTake();
        if (!value.has_value()) {
            return false;
        }
        out = std::move(*value);
        mImpl->Free(handle.mIndex);
        return true;
    }

    moe::Task<ReadbackLease> AsyncReadback::Read(const rhi::Buffer& src, uint64_t offset,
            uint64_t size, rhi::PipelineStage srcStage, rhi::Access srcAccess) {
        const ReadbackHandle handle = Request(src, offset, size, srcStage, srcAccess);
        if (!handle.IsValid()) {
            co_return ReadbackLease{};
        }
        ReadbackLease lease = co_await mImpl->mSlots[handle.mIndex]->mEvent;
        mImpl->Free(handle.mIndex);
        co_return lease;
    }
}// namespace moe::neo
