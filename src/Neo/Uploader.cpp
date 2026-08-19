#include "Neo/Uploader.hpp"

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
    }// namespace

    bool Uploader::Init(rhi::Device& device, std::string& error) {
        if (!device.WaitIdle()) {
            error = "Uploader: device WaitIdle failed";
            return false;
        }
        mDevice = &device;
        return true;
    }

    bool Uploader::UploadMesh(const Mesh& mesh, UploadedMesh& out, std::string& error) {
        if (mDevice == nullptr) {
            error = "Uploader: not initialized";
            return false;
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
            vertexBase += count;
        }

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
        rhi::BufferCreateInfo stagingInfo{};
        stagingInfo.mSize = vertexBytes + indexBytes;
        stagingInfo.mUsage = rhi::BufferUsage::kTransferSrc;
        stagingInfo.mCpuVisible = true;

        rhi::Buffer staging;
        if (!mDevice->CreateBuffer(vertexInfo, out.mVertexBuffer)
                || !mDevice->CreateBuffer(indexInfo, out.mIndexBuffer)
                || !mDevice->CreateBuffer(stagingInfo, staging)) {
            error = "Uploader: buffer creation failed: " + mDevice->GetLastError();
            return false;
        }

        {
            auto* data = static_cast<uint8_t*>(staging.Map());
            if (data == nullptr) {
                error = "Uploader: failed to map staging buffer";
                return false;
            }
            std::memcpy(data, vertexData.data(), vertexData.size());
            std::memcpy(data + vertexData.size(), indexData.data(), indexData.size() * sizeof(uint32_t));
            staging.Unmap();
        }

        rhi::CommandList cmd;
        if (!mDevice->CreateCommandList(cmd)) {
            error = "Uploader: command list creation failed";
            return false;
        }
        cmd.Begin();
        cmd.CopyBuffer(staging, out.mVertexBuffer, vertexBytes, 0, 0);
        cmd.CopyBuffer(staging, out.mIndexBuffer, indexBytes, vertexBytes, 0);
        cmd.End();
        if (!mDevice->Submit(cmd, true)) {
            error = "Uploader: submit failed: " + mDevice->GetLastError();
            cmd.Destroy();
            staging.Destroy();
            return false;
        }
        cmd.Destroy();
        staging.Destroy();

        out.mVertexCount = vertexBase;
        out.mIndexCount = static_cast<uint32_t>(indexData.size());
        out.mVertexStride = stride;
        out.mPositionOffset = 0;
        out.mNormalOffset = normalOffset;
        out.mUvOffset = uvOffset;
        out.mColorOffset = colorOffset;
        moe::Logger::info("Uploaded mesh '{}' ({} verts, {} indices, stride {})",
                mesh.mName, out.mVertexCount, out.mIndexCount, out.mVertexStride);
        return true;
    }

    bool Uploader::UploadTexture(const Texture& texture, UploadedTexture& out, std::string& error) {
        if (mDevice == nullptr) {
            error = "Uploader: not initialized";
            return false;
        }
        if (texture.mData.empty() || texture.mWidth == 0 || texture.mHeight == 0) {
            error = "Uploader: texture has no pixel data";
            return false;
        }
        // RGBA8 only (row pitch is tight and 4-aligned, so no staging tiling)
        if (texture.mChannels != 4) {
            error = "Uploader: only RGBA8 textures are supported";
            return false;
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
            error = "Uploader: image creation failed: " + mDevice->GetLastError();
            return false;
        }

        rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mMagFilter = rhi::Filter::kLinear;
        samplerInfo.mMinFilter = rhi::Filter::kLinear;
        if (!mDevice->CreateSampler(samplerInfo, out.mSampler)) {
            error = "Uploader: sampler creation failed: " + mDevice->GetLastError();
            out.mImage.Destroy();
            return false;
        }

        rhi::BufferCreateInfo stagingInfo{};
        stagingInfo.mSize = texture.mData.size();
        stagingInfo.mUsage = rhi::BufferUsage::kTransferSrc;
        stagingInfo.mCpuVisible = true;
        rhi::Buffer staging;
        if (!mDevice->CreateBuffer(stagingInfo, staging)) {
            error = "Uploader: staging buffer creation failed: " + mDevice->GetLastError();
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return false;
        }
        {
            auto* data = static_cast<uint8_t*>(staging.Map());
            if (data == nullptr) {
                error = "Uploader: failed to map staging buffer";
                staging.Destroy();
                out.mSampler.Destroy();
                out.mImage.Destroy();
                return false;
            }
            std::memcpy(data, texture.mData.data(), texture.mData.size());
            staging.Unmap();
        }

        rhi::CommandList cmd;
        if (!mDevice->CreateCommandList(cmd)) {
            error = "Uploader: command list creation failed";
            staging.Destroy();
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return false;
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
            error = "Uploader: submit failed: " + mDevice->GetLastError();
            cmd.Destroy();
            staging.Destroy();
            out.mSampler.Destroy();
            out.mImage.Destroy();
            return false;
        }
        cmd.Destroy();
        staging.Destroy();
        moe::Logger::info("Uploaded texture '{}' ({}x{}x{}, {} ch, {})",
                texture.mName, texture.mWidth, texture.mHeight, texture.mDepth,
                texture.mChannels, texture.mSrgb ? "sRGB" : "linear");
        return true;
    }

    bool Uploader::UpdateMeshVertices(const UploadedMesh& mesh,
            const uint8_t* vertexData, size_t byteCount, std::string& error) {
        if (mDevice == nullptr) {
            error = "Uploader: not initialized";
            return false;
        }
        if (vertexData == nullptr || byteCount == 0
                || byteCount > mesh.mVertexBuffer.GetSize()) {
            error = "Uploader: vertex update out of range";
            return false;
        }

        rhi::BufferCreateInfo stagingInfo{};
        stagingInfo.mSize = byteCount;
        stagingInfo.mUsage = rhi::BufferUsage::kTransferSrc;
        stagingInfo.mCpuVisible = true;
        rhi::Buffer staging;
        if (!mDevice->CreateBuffer(stagingInfo, staging)) {
            error = "Uploader: staging buffer creation failed: " + mDevice->GetLastError();
            return false;
        }
        {
            auto* data = static_cast<uint8_t*>(staging.Map());
            if (data == nullptr) {
                error = "Uploader: failed to map staging buffer";
                staging.Destroy();
                return false;
            }
            std::memcpy(data, vertexData, byteCount);
            staging.Unmap();
        }

        rhi::CommandList cmd;
        if (!mDevice->CreateCommandList(cmd)) {
            error = "Uploader: command list creation failed";
            staging.Destroy();
            return false;
        }
        cmd.Begin();
        cmd.CopyBuffer(staging, mesh.mVertexBuffer, byteCount, 0, 0);
        // staging write -> vertex shader read (next frame's draws)
        rhi::SyncInfo sync{};
        sync.mSrcStage = rhi::PipelineStage::kTransfer;
        sync.mSrcAccess = rhi::Access::kTransferWrite;
        sync.mDstStage = rhi::PipelineStage::kVertexShader;
        sync.mDstAccess = rhi::Access::kShaderRead;
        cmd.BufferBarrier(mesh.mVertexBuffer, sync);
        cmd.End();
        // async submit: staging and the command list are released through the
        // device's deferred-deletion queue once the GPU is done with them
        if (!mDevice->Submit(cmd, false)) {
            error = "Uploader: submit failed: " + mDevice->GetLastError();
            cmd.Destroy();
            staging.Destroy();
            return false;
        }
        cmd.Destroy();
        staging.Destroy();
        return true;
    }

    bool Uploader::UploadData(const uint8_t* data, size_t byteCount, rhi::BufferUsage usage,
            rhi::Buffer& out, std::string& error) {
        if (mDevice == nullptr) {
            error = "Uploader: not initialized";
            return false;
        }
        if (byteCount == 0 || data == nullptr) {
            error = "UploadData: empty data";
            return false;
        }

        rhi::BufferCreateInfo dstInfo{};
        dstInfo.mSize = byteCount;
        dstInfo.mUsage = usage | rhi::BufferUsage::kTransferDst;
        if (!mDevice->CreateBuffer(dstInfo, out)) {
            error = "UploadData: buffer: " + mDevice->GetLastError();
            return false;
        }

        rhi::BufferCreateInfo stagingInfo{};
        stagingInfo.mSize = byteCount;
        stagingInfo.mUsage = rhi::BufferUsage::kTransferSrc;
        stagingInfo.mCpuVisible = true;
        rhi::Buffer staging;
        if (!mDevice->CreateBuffer(stagingInfo, staging)) {
            error = "UploadData: staging: " + mDevice->GetLastError();
            out.Destroy();
            return false;
        }
        void* mapped = staging.Map();
        if (mapped == nullptr) {
            error = "UploadData: staging map failed";
            staging.Destroy();
            out.Destroy();
            return false;
        }
        std::memcpy(mapped, data, byteCount);
        staging.Unmap();

        rhi::CommandList cmd;
        if (!mDevice->CreateCommandList(cmd)) {
            error = "UploadData: command list creation failed";
            staging.Destroy();
            out.Destroy();
            return false;
        }
        cmd.Begin();
        cmd.CopyBuffer(staging, out, byteCount, 0, 0);
        rhi::SyncInfo sync{};
        sync.mSrcStage = rhi::PipelineStage::kTransfer;
        sync.mSrcAccess = rhi::Access::kTransferWrite;
        sync.mDstStage = rhi::PipelineStage::kVertexShader;
        sync.mDstAccess = rhi::Access::kShaderRead;
        cmd.BufferBarrier(out, sync);
        cmd.End();
        if (!mDevice->Submit(cmd, true)) {
            error = "UploadData: submit failed: " + mDevice->GetLastError();
            cmd.Destroy();
            staging.Destroy();
            out.Destroy();
            return false;
        }
        cmd.Destroy();
        staging.Destroy();
        moe::Logger::info("Uploaded buffer ({} bytes)", byteCount);
        return true;
    }
}// namespace moe::neo