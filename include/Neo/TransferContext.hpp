#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/Device.hpp>
#include <RHI/RHICommon.hpp>
#include <RHI/TimelineSemaphore.hpp>

#include <cstdint>
#include <functional>
#include <memory>

namespace moe::neo {
    using TransferSlotId = uint32_t;
    constexpr TransferSlotId kInvalidTransferSlot = UINT32_MAX;

    // Shared GPU transfer machinery. Owns a pool of host-visible staging
    // buffers and one timeline semaphore, plus a dedicated thread that waits on
    // the semaphore for GPU completion. Uploads (CPU->GPU) and readbacks
    // (GPU->CPU) both route through it. All recording/submission happens on the
    // calling (main) thread; only completion notification runs off it.
    //
    // Lifecycle is explicit: Init() then Shutdown() (the destructor is a leak
    // trap). Shutdown() joins the completion thread; call it before destroying
    // the Device and after any AsyncReadback built on this context.
    class TransferContext {
    public:
        TransferContext();
        ~TransferContext();

        TransferContext(const TransferContext&) = delete;
        TransferContext& operator=(const TransferContext&) = delete;

        bool Init(rhi::Device& device);
        void Shutdown();

        // CPU->GPU: staging write + copy into dst, with a transfer->dstStage
        // barrier. waitForCompletion blocks on the timeline (Uploader's
        // synchronous paths); otherwise the staging slot is recycled by the
        // completion thread once the GPU is done.
        bool Upload(const uint8_t* data, size_t byteCount, const rhi::Buffer& dst,
                rhi::PipelineStage dstStage, rhi::Access dstAccess, bool waitForCompletion);

        // GPU->CPU: records a copy of src[offset, offset+size) into a staging
        // slot, submitted by the next Pump(). onComplete(slot, mapped, size)
        // runs on the completion thread once the copy finishes; the slot stays
        // mapped and owned by the caller until ReleaseSlot. srcStage/srcAccess
        // describe the last write to src (default: a compute storage write).
        using ReadbackCallback = std::function<void(TransferSlotId, std::byte*, uint64_t)>;
        bool EnqueueReadback(const rhi::Buffer& src, uint64_t offset, uint64_t size,
                rhi::PipelineStage srcStage, rhi::Access srcAccess, ReadbackCallback onComplete);

        // Maps a staging slot for CPU access (host-visible, cache invalidated).
        std::byte* MapSlot(TransferSlotId slot);
        // Unmaps and returns the slot to the pool.
        void ReleaseSlot(TransferSlotId slot);

        // Submits queued readback copies. Call once per frame after Present.
        void Pump();

        // Blocks until every submitted copy has been processed. The GPU must be
        // idle first (e.g. after Device::WaitIdle). Used at shutdown.
        void Drain();

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
