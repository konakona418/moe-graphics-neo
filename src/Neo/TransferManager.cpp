#include "Neo/TransferManager.hpp"
#include <Core/Profile.hpp>

#include <Core/AsyncEvent.hpp>
#include <Core/Error.hpp>
#include <Core/Logger.hpp>
#include <RHI/CommandList.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace moe::neo {
    namespace {
        void AppendVec3(std::vector<uint8_t>& out, const glm::vec3& v) {
            const size_t base = out.size();
            out.resize(base + sizeof(float) * 3);
            std::memcpy(out.data() + base, &v, sizeof(float) * 3);
        }

        void AppendVec2(std::vector<uint8_t>& out, const glm::vec2& v) {
            const size_t base = out.size();
            out.resize(base + sizeof(float) * 2);
            std::memcpy(out.data() + base, &v, sizeof(float) * 2);
        }

        // Interleaves one primitive's attributes (layout chosen by the presence
        // flags) and appends its indices, rebased on vertexBase.
        void PackPrimitive(const MeshPrimitive& primitive, bool hasNormals, bool hasUvs,
                bool hasColors, std::vector<uint8_t>& vertexData,
                std::vector<uint32_t>& indexData, uint32_t vertexBase) {
            constexpr uint32_t kRgba8Size = 4;
            const uint32_t count = static_cast<uint32_t>(primitive.mPositions.size());
            for (uint32_t i = 0; i < count; ++i) {
                AppendVec3(vertexData, primitive.mPositions[i]);
                if (hasNormals) {
                    const glm::vec3 normal = i < primitive.mNormals.size()
                            ? primitive.mNormals[i] : glm::vec3(0.0f, 0.0f, 1.0f);
                    AppendVec3(vertexData, normal);
                }
                if (hasUvs) {
                    const glm::vec2 uv = i < primitive.mUv0.size()
                            ? primitive.mUv0[i] : glm::vec2(0.0f, 0.0f);
                    AppendVec2(vertexData, uv);
                }
                if (hasColors) {
                    const glm::vec4 color = i < primitive.mColors.size()
                            ? primitive.mColors[i] : glm::vec4(1.0f);
                    const uint8_t rgba[4] = {
                            static_cast<uint8_t>(glm::clamp(color.r, 0.0f, 1.0f) * 255.0f),
                            static_cast<uint8_t>(glm::clamp(color.g, 0.0f, 1.0f) * 255.0f),
                            static_cast<uint8_t>(glm::clamp(color.b, 0.0f, 1.0f) * 255.0f),
                            static_cast<uint8_t>(glm::clamp(color.a, 0.0f, 1.0f) * 255.0f),
                    };
                    const size_t base = vertexData.size();
                    vertexData.resize(base + kRgba8Size);
                    std::memcpy(vertexData.data() + base, rgba, kRgba8Size);
                }
            }
            for (const uint32_t index : primitive.mIndices) {
                indexData.push_back(vertexBase + index);
            }
        }

        struct ReadbackRequest {
            explicit ReadbackRequest(moe::Scheduler& scheduler) : mEvent(scheduler) {}

            moe::AsyncEvent<ReadbackLease> mEvent;
            TransferSlotId mSlot{kInvalidTransferSlot};
            uint32_t mGeneration{0};
            bool mActive{false};
        };
    }// namespace

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

    struct TransferManager::Impl {
        rhi::Device* mDevice{nullptr};
        moe::Scheduler* mScheduler{nullptr};
        TransferContext mContext;
        std::vector<std::shared_ptr<ReadbackRequest>> mRequests;
        std::vector<uint32_t> mFree;
        bool mRunning{false};

        uint32_t AllocRequest() {
            if (!mFree.empty()) {
                const uint32_t index = mFree.back();
                mFree.pop_back();
                return index;
            }
            mRequests.push_back(std::make_shared<ReadbackRequest>(*mScheduler));
            return static_cast<uint32_t>(mRequests.size() - 1);
        }

        void FreeRequest(uint32_t index) {
            mRequests[index]->mActive = false;
            mRequests[index]->mGeneration++;
            mFree.push_back(index);
        }
    };

    TransferManager::TransferManager() = default;

    TransferManager::~TransferManager() {
        Shutdown();
    }

    bool TransferManager::Init(rhi::Device& device, moe::Scheduler& scheduler) {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mRunning) {
            return false;
        }
        if (mImpl == nullptr) {
            mImpl = std::make_unique<Impl>();
        }
        mImpl->mDevice = &device;
        mImpl->mScheduler = &scheduler;
        if (!mImpl->mContext.Init(device)) {
            return false;
        }
        mImpl->mRunning = true;
        return true;
    }

    void TransferManager::Shutdown() {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return;
        }
        // Drain in-flight readbacks so no completion callback outlives its
        // request. The GPU must be idle first (e.g. after Device::WaitIdle).
        mImpl->mContext.Drain();
        mImpl->mRequests.clear();
        mImpl->mFree.clear();
        mImpl->mContext.Shutdown();
        mImpl->mRunning = false;
    }

    void TransferManager::Pump() {
        if (mImpl != nullptr) {
            mImpl->mContext.Pump();
        }
    }

    // ---- CPU -> GPU ----

    bool TransferManager::UploadMesh(const Mesh& mesh, UploadedMesh& out) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferManager: not initialized");
        }

        bool hasNormals = false;
        bool hasUvs = false;
        bool hasColors = false;
        for (const auto& primitive : mesh.mPrimitives) {
            hasNormals = hasNormals || !primitive.mNormals.empty();
            hasUvs = hasUvs || !primitive.mUv0.empty();
            hasColors = hasColors || !primitive.mColors.empty();
        }
        constexpr uint32_t kVec3Size = 12;
        constexpr uint32_t kVec2Size = 8;
        constexpr uint32_t kRgba8Size = 4;
        const uint32_t stride = kVec3Size + (hasNormals ? kVec3Size : 0) + (hasUvs ? kVec2Size : 0)
                + (hasColors ? kRgba8Size : 0);
        const uint32_t normalOffset = hasNormals ? kVec3Size : stride;
        const uint32_t uvOffset = kVec3Size + (hasNormals ? kVec3Size : 0);
        const uint32_t colorOffset = uvOffset + (hasUvs ? kVec2Size : 0);

        std::vector<uint8_t> vertexData;
        std::vector<uint32_t> indexData;
        uint32_t vertexBase = 0;
        for (const auto& primitive : mesh.mPrimitives) {
            const uint32_t count = static_cast<uint32_t>(primitive.mPositions.size());
            PackPrimitive(primitive, hasNormals, hasUvs, hasColors, vertexData, indexData, vertexBase);
            vertexBase += count;
        }

        if (!UploadMeshData(vertexData, indexData, stride, normalOffset, uvOffset, colorOffset, out)) {
            return false;
        }
        moe::Logger::Info("Uploaded mesh '{}' ({} verts, {} indices, stride {})",
                mesh.mName, out.mVertexCount, out.mIndexCount, out.mVertexStride);
        return true;
    }

    bool TransferManager::UploadMeshPrimitive(const MeshPrimitive& primitive, UploadedMesh& out) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferManager: not initialized");
        }

        const bool hasNormals = !primitive.mNormals.empty();
        const bool hasUvs = !primitive.mUv0.empty();
        const bool hasColors = !primitive.mColors.empty();
        constexpr uint32_t kVec3Size = 12;
        constexpr uint32_t kVec2Size = 8;
        constexpr uint32_t kRgba8Size = 4;
        const uint32_t stride = kVec3Size + (hasNormals ? kVec3Size : 0) + (hasUvs ? kVec2Size : 0)
                + (hasColors ? kRgba8Size : 0);
        const uint32_t normalOffset = hasNormals ? kVec3Size : stride;
        const uint32_t uvOffset = kVec3Size + (hasNormals ? kVec3Size : 0);
        const uint32_t colorOffset = uvOffset + (hasUvs ? kVec2Size : 0);

        std::vector<uint8_t> vertexData;
        std::vector<uint32_t> indexData;
        PackPrimitive(primitive, hasNormals, hasUvs, hasColors, vertexData, indexData, 0);

        if (!UploadMeshData(vertexData, indexData, stride, normalOffset, uvOffset, colorOffset, out)) {
            return false;
        }
        moe::Logger::Info("Uploaded mesh primitive ({} verts, {} indices, stride {})",
                out.mVertexCount, out.mIndexCount, out.mVertexStride);
        return true;
    }

    bool TransferManager::UploadMeshData(const std::vector<uint8_t>& vertexData,
            const std::vector<uint32_t>& indexData, uint32_t stride,
            uint32_t normalOffset, uint32_t uvOffset, uint32_t colorOffset,
            UploadedMesh& out) {
        MOE_PROFILE_ZONE();
        const uint64_t vertexBytes = vertexData.size();
        const uint64_t indexBytes = indexData.size() * sizeof(uint32_t);

        rhi::BufferCreateInfo vertexInfo{};
        vertexInfo.mSize = vertexBytes;
        vertexInfo.mUsage = rhi::BufferUsage::kVertex | rhi::BufferUsage::kTransferDst
                | rhi::BufferUsage::kTransferSrc;
        rhi::BufferCreateInfo indexInfo{};
        indexInfo.mSize = indexBytes;
        indexInfo.mUsage = rhi::BufferUsage::kIndex | rhi::BufferUsage::kTransferDst
                | rhi::BufferUsage::kTransferSrc;

        if (!mImpl->mDevice->CreateBuffer(vertexInfo, out.mVertexBuffer)
                || !mImpl->mDevice->CreateBuffer(indexInfo, out.mIndexBuffer)) {
            out.mVertexBuffer.Destroy();
            out.mIndexBuffer.Destroy();
            return false;
        }
        if (!mImpl->mContext.Upload(vertexData.data(), vertexBytes, out.mVertexBuffer,
                    rhi::PipelineStage::kVertexInput, rhi::Access::kVertexAttributeRead, true)
                || !mImpl->mContext.Upload(reinterpret_cast<const uint8_t*>(indexData.data()),
                        indexBytes, out.mIndexBuffer, rhi::PipelineStage::kVertexInput,
                        rhi::Access::kIndexRead, true)) {
            out.mVertexBuffer.Destroy();
            out.mIndexBuffer.Destroy();
            return false;
        }

        out.mVertexCount = static_cast<uint32_t>(vertexData.size() / stride);
        out.mIndexCount = static_cast<uint32_t>(indexData.size());
        out.mVertexStride = stride;
        out.mPositionOffset = 0;
        out.mNormalOffset = normalOffset;
        out.mUvOffset = uvOffset;
        out.mColorOffset = colorOffset;
        return true;
    }

    bool TransferManager::UpdateMeshVertices(const UploadedMesh& mesh,
            const uint8_t* vertexData, size_t byteCount) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferManager: not initialized");
        }
        if (vertexData == nullptr || byteCount == 0
                || byteCount > mesh.mVertexBuffer.GetSize()) {
            return moe::Fail("TransferManager: vertex update out of range");
        }
        return mImpl->mContext.Upload(vertexData, byteCount, mesh.mVertexBuffer,
                rhi::PipelineStage::kVertexInput, rhi::Access::kVertexAttributeRead, false);
    }

    bool TransferManager::UploadTexture(const Texture& texture, UploadedTexture& out) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferManager: not initialized");
        }
        if (texture.mData.empty() || texture.mWidth == 0 || texture.mHeight == 0) {
            return moe::Fail("TransferManager: texture has no pixel data");
        }
        // RGBA8 only (row pitch is tight and 4-aligned, so no staging tiling)
        if (texture.mChannels != 4) {
            return moe::Fail("TransferManager: only RGBA8 textures are supported");
        }

        rhi::ImageCreateInfo imageInfo{};
        imageInfo.mType = texture.mDepth > 1 ? rhi::ImageType::k3D : rhi::ImageType::k2D;
        imageInfo.mWidth = texture.mWidth;
        imageInfo.mHeight = texture.mHeight;
        imageInfo.mDepth = texture.mDepth;
        imageInfo.mMipLevels = texture.mMipLevels;
        imageInfo.mFormat = texture.mSrgb ? rhi::Format::kR8G8B8A8Srgb : rhi::Format::kR8G8B8A8Unorm;
        imageInfo.mUsage = rhi::ImageUsage::kSampled | rhi::ImageUsage::kTransferDst
                | rhi::ImageUsage::kTransferSrc; // kTransferSrc: readback/debug
        if (!mImpl->mDevice->CreateImage(imageInfo, out.mImage)) {
            return moe::Fail("TransferManager: image creation failed: " + moe::Error::Get());
        }

        rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mMagFilter = rhi::Filter::kLinear;
        samplerInfo.mMinFilter = rhi::Filter::kLinear;
        if (!mImpl->mDevice->CreateSampler(samplerInfo, out.mSampler)) {
            out.mImage.Destroy();
            return moe::Fail("TransferManager: sampler creation failed: " + moe::Error::Get());
        }

        const TransferSlotId slot = mImpl->mContext.AcquireStaging(texture.mData.size());
        if (slot == kInvalidTransferSlot) {
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return moe::Fail("TransferManager: staging allocation failed");
        }
        {
            std::byte* data = mImpl->mContext.MapSlot(slot);
            if (data == nullptr) {
                mImpl->mContext.ReleaseSlot(slot);
                out.mSampler.Destroy();
                out.mImage.Destroy();
                return moe::Fail("TransferManager: failed to map staging buffer");
            }
            std::memcpy(data, texture.mData.data(), texture.mData.size());
            mImpl->mContext.UnmapSlot(slot);
        }

        rhi::CommandList cmd;
        if (!mImpl->mDevice->CreateCommandList(cmd)) {
            mImpl->mContext.ReleaseSlot(slot);
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return moe::Fail("TransferManager: command list creation failed");
        }
        cmd.Begin();

        rhi::SyncInfo toTransfer{};
        toTransfer.mSrcStage = rhi::PipelineStage::kTopOfPipe;
        toTransfer.mSrcAccess = rhi::Access::kNone;
        toTransfer.mDstStage = rhi::PipelineStage::kTransfer;
        toTransfer.mDstAccess = rhi::Access::kTransferWrite;
        cmd.ImageBarrier(out.mImage, rhi::ImageLayout::kUndefined,
                rhi::ImageLayout::kTransferDst, toTransfer);

        // Texture::mData packs all mip levels tightly, level 0 first; each
        // level's extent is the base extent shifted down by its index.
        const rhi::Buffer& staging = mImpl->mContext.GetStagingBuffer(slot);
        const uint32_t levels = texture.mMipLevels;
        size_t offset = 0;
        bool packedOk = true;
        for (uint32_t level = 0; level < levels; ++level) {
            const uint32_t width = std::max(1u, texture.mWidth >> level);
            const uint32_t height = std::max(1u, texture.mHeight >> level);
            const uint32_t depth = std::max(1u, texture.mDepth >> level);
            const size_t levelBytes = static_cast<size_t>(width) * height * depth
                    * texture.mChannels;
            if (offset + levelBytes > texture.mData.size()) {
                packedOk = false;
                break;
            }
            cmd.CopyBufferToImage(staging, out.mImage, level, 0, 1,
                    static_cast<uint32_t>(offset));
            offset += levelBytes;
        }
        if (!packedOk) {
            cmd.Destroy();
            mImpl->mContext.ReleaseSlot(slot);
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return moe::Fail("TransferManager: texture data smaller than its mip chain");
        }

        rhi::SyncInfo toSample{};
        toSample.mSrcStage = rhi::PipelineStage::kTransfer;
        toSample.mSrcAccess = rhi::Access::kTransferWrite;
        toSample.mDstStage = rhi::PipelineStage::kFragmentShader;
        toSample.mDstAccess = rhi::Access::kShaderRead;
        cmd.ImageBarrier(out.mImage, rhi::ImageLayout::kTransferDst,
                rhi::ImageLayout::kShaderReadOnly, toSample);

        cmd.End();
        if (!mImpl->mDevice->Submit(cmd, true)) {
            cmd.Destroy();
            mImpl->mContext.ReleaseSlot(slot);
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return moe::Fail("TransferManager: submit failed: " + moe::Error::Get());
        }
        cmd.Destroy();
        mImpl->mContext.ReleaseSlot(slot);
        moe::Logger::Info("Uploaded texture '{}' ({}x{}x{}, {} ch, {})",
                texture.mName, texture.mWidth, texture.mHeight, texture.mDepth,
                texture.mChannels, texture.mSrgb ? "sRGB" : "linear");
        return true;
    }

    bool TransferManager::UploadData(const uint8_t* data, size_t byteCount, rhi::BufferUsage usage,
            rhi::Buffer& out, rhi::PipelineStage dstStage, rhi::Access dstAccess) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferManager: not initialized");
        }
        if (byteCount == 0 || data == nullptr) {
            return moe::Fail("UploadData: empty data");
        }

        rhi::BufferCreateInfo dstInfo{};
        dstInfo.mSize = byteCount;
        dstInfo.mUsage = usage | rhi::BufferUsage::kTransferDst;
        if (!mImpl->mDevice->CreateBuffer(dstInfo, out)) {
            return moe::Fail("UploadData: buffer: " + moe::Error::Get());
        }
        if (!mImpl->mContext.Upload(data, byteCount, out, dstStage, dstAccess, true)) {
            out.Destroy();
            return false;
        }
        moe::Logger::Info("Uploaded buffer ({} bytes)", byteCount);
        return true;
    }

    bool TransferManager::UpdateBuffer(const rhi::Buffer& dst, const void* data, size_t byteCount,
            rhi::PipelineStage dstStage, rhi::Access dstAccess) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return moe::Fail("TransferManager: not initialized");
        }
        if (byteCount == 0 || data == nullptr || byteCount > dst.GetSize()) {
            return moe::Fail("UpdateBuffer: empty or out of range");
        }
        return mImpl->mContext.Upload(static_cast<const uint8_t*>(data), byteCount, dst,
                dstStage, dstAccess, true);
    }

    // ---- GPU -> CPU ----

    ReadbackHandle TransferManager::Request(const rhi::Buffer& src, uint64_t offset, uint64_t size,
            rhi::PipelineStage srcStage, rhi::Access srcAccess) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            moe::Error::Set("TransferManager: not initialized");
            return {};
        }
        const uint32_t index = mImpl->AllocRequest();
        const std::shared_ptr<ReadbackRequest> request = mImpl->mRequests[index];
        request->mActive = true;
        request->mSlot = kInvalidTransferSlot;

        TransferContext* context = &mImpl->mContext;
        const bool ok = context->EnqueueReadback(src, offset, size, srcStage, srcAccess,
                [request, context](TransferSlotId slot, std::byte* mapped, uint64_t bytes) {
                    request->mSlot = slot;
                    const bool valid = mapped != nullptr;
                    const std::span<std::byte> span = valid
                            ? std::span<std::byte>(mapped, bytes)
                            : std::span<std::byte>{};
                    request->mEvent.SetValue(
                            ReadbackLease{span, valid ? context : nullptr, slot});
                });
        if (!ok) {
            mImpl->FreeRequest(index);
            return {};
        }
        return ReadbackHandle{index, request->mGeneration};
    }

    bool TransferManager::TryConsume(ReadbackHandle handle, ReadbackLease& out) {
        if (mImpl == nullptr || !mImpl->mRunning || handle.mIndex >= mImpl->mRequests.size()) {
            return false;
        }
        const std::shared_ptr<ReadbackRequest> request = mImpl->mRequests[handle.mIndex];
        if (!request->mActive || request->mGeneration != handle.mGeneration) {
            return false;
        }
        std::optional<ReadbackLease> value = request->mEvent.TryTake();
        if (!value.has_value()) {
            return false;
        }
        out = std::move(*value);
        mImpl->FreeRequest(handle.mIndex);
        return true;
    }

    moe::Task<ReadbackLease> TransferManager::Read(const rhi::Buffer& src, uint64_t offset,
            uint64_t size, rhi::PipelineStage srcStage, rhi::Access srcAccess) {
        const ReadbackHandle handle = Request(src, offset, size, srcStage, srcAccess);
        if (!handle.IsValid()) {
            co_return ReadbackLease{};
        }
        ReadbackLease lease = co_await mImpl->mRequests[handle.mIndex]->mEvent;
        mImpl->FreeRequest(handle.mIndex);
        co_return lease;
    }
}// namespace moe::neo
