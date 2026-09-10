#include "Neo/Uploader.hpp"

#include <Core/Error.hpp>
#include <Core/Logger.hpp>
#include <RHI/CommandList.hpp>

#include <cstring>
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

        // Interleaves one primitive's attributes (layout chosen by the
        // presence flags) and appends its indices, rebased on vertexBase.
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
    }// namespace

    bool Uploader::Init(rhi::Device& device) {
        if (!device.WaitIdle()) {
            return moe::Fail("Uploader: device WaitIdle failed");
        }
        mDevice = &device;
        return true;
    }

    bool Uploader::CreateStagingBuffer(size_t size, rhi::Buffer& out) {
        rhi::BufferCreateInfo stagingInfo{};
        stagingInfo.mSize = size;
        stagingInfo.mUsage = rhi::BufferUsage::kTransferSrc;
        stagingInfo.mCpuVisible = true;
        if (!mDevice->CreateBuffer(stagingInfo, out)) {
            return moe::Fail("Uploader: staging buffer creation failed: " + moe::Error::Get());
        }
        return true;
    }

    bool Uploader::UploadBytes(const uint8_t* data, size_t byteCount, const rhi::Buffer& dst,
            bool waitForCompletion) {
        rhi::Buffer staging;
        if (!CreateStagingBuffer(byteCount, staging)) {
            return false;
        }
        {
            auto* mapped = static_cast<uint8_t*>(staging.Map());
            if (mapped == nullptr) {
                staging.Destroy();
                return moe::Fail("Uploader: failed to map staging buffer");
            }
            std::memcpy(mapped, data, byteCount);
            staging.Unmap();
        }

        rhi::CommandList cmd;
        if (!mDevice->CreateCommandList(cmd)) {
            staging.Destroy();
            return moe::Fail("Uploader: command list creation failed");
        }
        cmd.Begin();
        cmd.CopyBuffer(staging, dst, byteCount, 0, 0);
        // staging write -> shader read (this frame's or next frame's draws)
        rhi::SyncInfo sync{};
        sync.mSrcStage = rhi::PipelineStage::kTransfer;
        sync.mSrcAccess = rhi::Access::kTransferWrite;
        sync.mDstStage = rhi::PipelineStage::kVertexShader;
        sync.mDstAccess = rhi::Access::kShaderRead;
        cmd.BufferBarrier(dst, sync);
        cmd.End();
        if (!mDevice->Submit(cmd, waitForCompletion)) {
            cmd.Destroy();
            staging.Destroy();
            return moe::Fail("Uploader: submit failed: " + moe::Error::Get());
        }
        cmd.Destroy();
        staging.Destroy();
        return true;
    }

    bool Uploader::UploadMesh(const Mesh& mesh, UploadedMesh& out) {
        if (mDevice == nullptr) {
            return moe::Fail("Uploader: not initialized");
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
        moe::Logger::info("Uploaded mesh '{}' ({} verts, {} indices, stride {})",
                mesh.mName, out.mVertexCount, out.mIndexCount, out.mVertexStride);
        return true;
    }

    bool Uploader::UploadMeshPrimitive(const MeshPrimitive& primitive, UploadedMesh& out) {
        if (mDevice == nullptr) {
            return moe::Fail("Uploader: not initialized");
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
        moe::Logger::info("Uploaded mesh primitive ({} verts, {} indices, stride {})",
                out.mVertexCount, out.mIndexCount, out.mVertexStride);
        return true;
    }

    bool Uploader::UploadMeshData(const std::vector<uint8_t>& vertexData,
            const std::vector<uint32_t>& indexData, uint32_t stride,
            uint32_t normalOffset, uint32_t uvOffset, uint32_t colorOffset,
            UploadedMesh& out) {
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

        rhi::Buffer staging;
        if (!mDevice->CreateBuffer(vertexInfo, out.mVertexBuffer)
                || !mDevice->CreateBuffer(indexInfo, out.mIndexBuffer)
                || !CreateStagingBuffer(vertexBytes + indexBytes, staging)) {
            return false;
        }

        {
            auto* data = static_cast<uint8_t*>(staging.Map());
            if (data == nullptr) {
                out.mVertexBuffer.Destroy();
                out.mIndexBuffer.Destroy();
                staging.Destroy();
                return moe::Fail("Uploader: failed to map staging buffer");
            }
            std::memcpy(data, vertexData.data(), vertexData.size());
            std::memcpy(data + vertexData.size(), indexData.data(), indexData.size() * sizeof(uint32_t));
            staging.Unmap();
        }

        rhi::CommandList cmd;
        if (!mDevice->CreateCommandList(cmd)) {
            return moe::Fail("Uploader: command list creation failed");
        }
        cmd.Begin();
        cmd.CopyBuffer(staging, out.mVertexBuffer, vertexBytes, 0, 0);
        cmd.CopyBuffer(staging, out.mIndexBuffer, indexBytes, vertexBytes, 0);
        cmd.End();
        if (!mDevice->Submit(cmd, true)) {
            cmd.Destroy();
            staging.Destroy();
            return moe::Fail("Uploader: submit failed: " + moe::Error::Get());
        }
        cmd.Destroy();
        staging.Destroy();

        out.mVertexCount = static_cast<uint32_t>(vertexData.size() / stride);
        out.mIndexCount = static_cast<uint32_t>(indexData.size());
        out.mVertexStride = stride;
        out.mPositionOffset = 0;
        out.mNormalOffset = normalOffset;
        out.mUvOffset = uvOffset;
        out.mColorOffset = colorOffset;
        return true;
    }

    bool Uploader::UploadTexture(const Texture& texture, UploadedTexture& out) {
        if (mDevice == nullptr) {
            return moe::Fail("Uploader: not initialized");
        }
        if (texture.mData.empty() || texture.mWidth == 0 || texture.mHeight == 0) {
            return moe::Fail("Uploader: texture has no pixel data");
        }
        // RGBA8 only (row pitch is tight and 4-aligned, so no staging tiling)
        if (texture.mChannels != 4) {
            return moe::Fail("Uploader: only RGBA8 textures are supported");
        }

        rhi::ImageCreateInfo imageInfo{};
        imageInfo.mType = rhi::ImageType::k2D;
        imageInfo.mWidth = texture.mWidth;
        imageInfo.mHeight = texture.mHeight;
        imageInfo.mDepth = 1;
        imageInfo.mMipLevels = texture.mMipLevels;
        imageInfo.mFormat = texture.mSrgb ? rhi::Format::kR8G8B8A8Srgb : rhi::Format::kR8G8B8A8Unorm;
        imageInfo.mUsage = rhi::ImageUsage::kSampled | rhi::ImageUsage::kTransferDst
                | rhi::ImageUsage::kTransferSrc; // kTransferSrc: readback/debug
        if (!mDevice->CreateImage(imageInfo, out.mImage)) {
            return moe::Fail("Uploader: image creation failed: " + moe::Error::Get());
        }

        rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mMagFilter = rhi::Filter::kLinear;
        samplerInfo.mMinFilter = rhi::Filter::kLinear;
        if (!mDevice->CreateSampler(samplerInfo, out.mSampler)) {
            out.mImage.Destroy();
            return moe::Fail("Uploader: sampler creation failed: " + moe::Error::Get());
        }

        rhi::Buffer staging;
        if (!CreateStagingBuffer(texture.mData.size(), staging)) {
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return false;
        }
        {
            auto* data = static_cast<uint8_t*>(staging.Map());
            if (data == nullptr) {
                staging.Destroy();
                out.mSampler.Destroy();
                out.mImage.Destroy();
                return moe::Fail("Uploader: failed to map staging buffer");
            }
            std::memcpy(data, texture.mData.data(), texture.mData.size());
            staging.Unmap();
        }

        rhi::CommandList cmd;
        if (!mDevice->CreateCommandList(cmd)) {
            staging.Destroy();
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return moe::Fail("Uploader: command list creation failed");
        }
        cmd.Begin();

        rhi::SyncInfo toTransfer{};
        toTransfer.mSrcStage = rhi::PipelineStage::kTopOfPipe;
        toTransfer.mSrcAccess = rhi::Access::kNone;
        toTransfer.mDstStage = rhi::PipelineStage::kTransfer;
        toTransfer.mDstAccess = rhi::Access::kTransferWrite;
        cmd.ImageBarrier(out.mImage, rhi::ImageLayout::kUndefined,
                rhi::ImageLayout::kTransferDst, toTransfer);

        cmd.CopyBufferToImage(staging, out.mImage, 0, 0, 1);

        rhi::SyncInfo toSample{};
        toSample.mSrcStage = rhi::PipelineStage::kTransfer;
        toSample.mSrcAccess = rhi::Access::kTransferWrite;
        toSample.mDstStage = rhi::PipelineStage::kFragmentShader;
        toSample.mDstAccess = rhi::Access::kShaderRead;
        cmd.ImageBarrier(out.mImage, rhi::ImageLayout::kTransferDst,
                rhi::ImageLayout::kShaderReadOnly, toSample);

        cmd.End();
        if (!mDevice->Submit(cmd, true)) {
            cmd.Destroy();
            staging.Destroy();
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return moe::Fail("Uploader: submit failed: " + moe::Error::Get());
        }
        cmd.Destroy();
        staging.Destroy();
        moe::Logger::info("Uploaded texture '{}' ({}x{}x{}, {} ch, {})",
                texture.mName, texture.mWidth, texture.mHeight, texture.mDepth,
                texture.mChannels, texture.mSrgb ? "sRGB" : "linear");
        return true;
    }

    bool Uploader::UpdateMeshVertices(const UploadedMesh& mesh,
            const uint8_t* vertexData, size_t byteCount) {
        if (mDevice == nullptr) {
            return moe::Fail("Uploader: not initialized");
        }
        if (vertexData == nullptr || byteCount == 0
                || byteCount > mesh.mVertexBuffer.GetSize()) {
            return moe::Fail("Uploader: vertex update out of range");
        }

        // async submit: staging and the command list are released through the
        // device's deferred-deletion queue once the GPU is done with them
        return UploadBytes(vertexData, byteCount, mesh.mVertexBuffer, false);
    }

    bool Uploader::UploadData(const uint8_t* data, size_t byteCount, rhi::BufferUsage usage,
            rhi::Buffer& out) {
        if (mDevice == nullptr) {
            return moe::Fail("Uploader: not initialized");
        }
        if (byteCount == 0 || data == nullptr) {
            return moe::Fail("UploadData: empty data");
        }

        rhi::BufferCreateInfo dstInfo{};
        dstInfo.mSize = byteCount;
        dstInfo.mUsage = usage | rhi::BufferUsage::kTransferDst;
        if (!mDevice->CreateBuffer(dstInfo, out)) {
            return moe::Fail("UploadData: buffer: " + moe::Error::Get());
        }
        if (!UploadBytes(data, byteCount, out, true)) {
            out.Destroy();
            return false;
        }
        moe::Logger::info("Uploaded buffer ({} bytes)", byteCount);
        return true;
    }
}// namespace moe::neo
