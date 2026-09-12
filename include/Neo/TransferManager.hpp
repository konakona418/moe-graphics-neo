#pragma once

#include <Core/Scheduler.hpp>
#include <Core/Task.hpp>
#include <Neo/Mesh.hpp>
#include <Neo/Texture.hpp>
#include <Neo/TransferContext.hpp>
#include <Neo/Uploaded.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/Device.hpp>
#include <RHI/RHICommon.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace moe::neo {
    // Move-only view into a leased staging slot. Reading is zero-copy: the span
    // points directly at the mapped staging buffer (cache-invalidated after the
    // GPU copy). Releasing — or simply dropping — the lease returns the slot to
    // the pool, so the lease must not outlive the TransferManager.
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

    // Unified GPU transfer service. Owns the staging pool, timeline semaphore,
    // and dedicated completion thread (TransferContext) and exposes both
    // directions: CPU->GPU uploads (the former Uploader API) and GPU->CPU
    // readbacks. Uploads are synchronous to callers; readbacks are polled
    // (TryConsume) or awaited (Read).
    //
    // Lifecycle is explicit: Init() then Shutdown() (the destructor is a leak
    // trap). Own it above the Device (e.g. examples::App) and inject it into
    // consumers, so Shutdown() runs before the Device is destroyed.
    class TransferManager {
    public:
        TransferManager();
        ~TransferManager();

        TransferManager(const TransferManager&) = delete;
        TransferManager& operator=(const TransferManager&) = delete;

        bool Init(rhi::Device& device, moe::Scheduler& scheduler);
        void Shutdown();

        // ---- CPU -> GPU ----

        // Uploads the whole mesh (all primitives concatenated; attribute layout
        // must be uniform across primitives).
        bool UploadMesh(const Mesh& mesh, UploadedMesh& out);

        // Uploads a single primitive into its own buffers. The content layer
        // uses this to preserve per-material primitive boundaries (UploadMesh
        // concatenates and would lose them).
        bool UploadMeshPrimitive(const MeshPrimitive& primitive, UploadedMesh& out);

        // Overwrites vertexData.size() bytes at the start of an uploaded vertex
        // buffer (staging write + transfer->vertex-read barrier). vertexData
        // must use the same interleaved layout as the upload.
        bool UpdateMeshVertices(const UploadedMesh& mesh,
                const uint8_t* vertexData, size_t byteCount);

        // Uploads texture pixels into a sampled image + linear sampler (sRGB
        // format when Texture::mSrgb is set). The image ends in ShaderReadOnly
        // layout; bind it with DescriptorSet::WriteImage. With mMipLevels > 1,
        // mData must contain the whole chain tightly packed (level 0 first).
        bool UploadTexture(const Texture& texture, UploadedTexture& out);

        // Uploads raw bytes into a device-local buffer (e.g. instance data).
        // The buffer is created with the given usage plus TransferDst.
        bool UploadData(const uint8_t* data, size_t byteCount, rhi::BufferUsage usage,
                rhi::Buffer& out,
                rhi::PipelineStage dstStage = rhi::PipelineStage::kVertexShader,
                rhi::Access dstAccess = rhi::Access::kShaderRead);

        // Overwrites the start of an existing device-local buffer through a
        // staging copy. Waits for completion, so the data is ready for the next
        // recorded draw.
        bool UpdateBuffer(const rhi::Buffer& dst, const void* data, size_t byteCount,
                rhi::PipelineStage dstStage = rhi::PipelineStage::kVertexInput,
                rhi::Access dstAccess = rhi::Access::kVertexAttributeRead);

        // ---- GPU -> CPU ----

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

        // Submits queued readback copies. Call once per frame after Present.
        void Pump();

    private:
        // Creates the vertex/index buffers and uploads packed data.
        bool UploadMeshData(const std::vector<uint8_t>& vertexData,
                const std::vector<uint32_t>& indexData, uint32_t stride,
                uint32_t normalOffset, uint32_t uvOffset, uint32_t colorOffset,
                UploadedMesh& out);

        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
