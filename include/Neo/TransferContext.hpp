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
    // the Device and after any TransferManager built on this context.
    class TransferContext {
    public:
        TransferContext();
        ~TransferContext();

        TransferContext(const TransferContext&) = delete;
        TransferContext& operator=(const TransferContext&) = delete;

        bool Init(rhi::Device& device,
                rhi::QueueType readbackQueue = rhi::QueueType::kCompute);
        void Shutdown();

        // CPU->GPU: staging write + copy into dst, with a transfer->dstStage
        // barrier. waitForCompletion blocks on the timeline (the upload path);
        // otherwise the staging slot is recycled by the completion thread once
        // the GPU is done.
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
        // Unmaps a staging slot (flushing CPU writes) without releasing it.
        void UnmapSlot(TransferSlotId slot);
        // Unmaps and returns the slot to the pool.
        void ReleaseSlot(TransferSlotId slot);

        // Raw staging access for callers that record their own transfer
        // commands (e.g. image/mip uploads). AcquireStaging returns a free slot
        // of at least `size` bytes; pair it with ReleaseSlot.
        TransferSlotId AcquireStaging(uint64_t size);
        rhi::Buffer& GetStagingBuffer(TransferSlotId slot);

        // Submits queued readback copies on the readback queue. Call once per
        // frame after Present. Uploads always run on the graphics queue (their
        // destinations are read by graphics, so no cross-queue hand-off).
        void Pump();

        // Blocks until every submitted copy has been processed. The GPU must be
        // idle first (e.g. after Device::WaitIdle). Used at shutdown.
        void Drain();

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
