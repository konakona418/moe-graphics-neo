#pragma once

#include <Core/Scheduler.hpp>
#include <Core/Task.hpp>
#include <Neo/TransferContext.hpp>
#include <RHI/RHICommon.hpp>

#include <cstdint>
#include <span>

namespace moe::neo {
    // Move-only view into a leased staging slot. Reading is zero-copy: the span
    // points directly at the mapped staging buffer (cache-invalidated after the
    // GPU copy). Releasing — or simply dropping — the lease returns the slot to
    // the TransferContext pool, so the lease must not outlive the context.
    class ReadbackLease {
    public:
        ReadbackLease() = default;
        ReadbackLease(std::span<std::byte> bytes, TransferContext* owner, TransferSlotId slot);
        ~ReadbackLease();

        ReadbackLease(ReadbackLease&& other) noexcept;
        ReadbackLease& operator=(ReadbackLease&& other) noexcept;
        ReadbackLease(const ReadbackLease&) = delete;
        ReadbackLease& operator=(const ReadbackLease&) = delete;

        std::span<std::byte> Bytes() const {
            return mBytes;
        }

        explicit operator bool() const {
            return mOwner != nullptr;
        }

        void Reset();

    private:
        std::span<std::byte> mBytes{};
        TransferContext* mOwner{nullptr};
        TransferSlotId mSlot{kInvalidTransferSlot};
    };

    struct ReadbackHandle {
        uint32_t mIndex{UINT32_MAX};
        uint32_t mGeneration{0};

        bool IsValid() const {
            return mIndex != UINT32_MAX;
        }
    };

    // GPU->CPU readback on top of a TransferContext. Request() enqueues a copy
    // (submitted by TransferContext::Pump), TryConsume() polls for the result,
    // and Read() exposes the same thing as a co_await-able Task.
    //
    // Shutdown() drains in-flight readbacks; do not destroy the AsyncReadback
    // (or the TransferContext) while a Read() coroutine is suspended.
    class AsyncReadback {
    public:
        AsyncReadback();
        ~AsyncReadback();

        AsyncReadback(const AsyncReadback&) = delete;
        AsyncReadback& operator=(const AsyncReadback&) = delete;

        bool Init(moe::Scheduler& scheduler, TransferContext& transfer);
        void Shutdown();

        // Enqueues a readback of src[offset, offset+size). srcStage/srcAccess
        // describe the last write to src (default: a compute storage write).
        ReadbackHandle Request(const rhi::Buffer& src, uint64_t offset, uint64_t size,
                rhi::PipelineStage srcStage = rhi::PipelineStage::kComputeShader,
                rhi::Access srcAccess = rhi::Access::kShaderWrite);

        // Moves the finished data out if ready. Returns false while pending.
        bool TryConsume(ReadbackHandle handle, ReadbackLease& out);

        // Coroutine form: co_await the result.
        moe::Task<ReadbackLease> Read(const rhi::Buffer& src, uint64_t offset, uint64_t size,
                rhi::PipelineStage srcStage = rhi::PipelineStage::kComputeShader,
                rhi::Access srcAccess = rhi::Access::kShaderWrite);

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
