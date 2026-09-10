#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/Sampler.hpp>

#include "Neo/Mesh.hpp"
#include "Neo/Texture.hpp"

#include <cstdint>
#include <vector>

namespace moe::neo {
    // GPU resources for one uploaded mesh. Vertex data is interleaved
    // position (+ normal + uv when present). Destroy() releases the RHI
    // buffers (deferred deletion; the RHI leak trap fires if forgotten).
    struct UploadedMesh {
        rhi::Buffer mVertexBuffer;
        rhi::Buffer mIndexBuffer;
        uint32_t mVertexCount{0};
        uint32_t mIndexCount{0};
        uint32_t mVertexStride{0};
        uint32_t mPositionOffset{0};
        uint32_t mNormalOffset{0}; // == stride when absent
        uint32_t mUvOffset{0};     // == stride when absent
        uint32_t mColorOffset{0};  // == stride when absent

        bool HasNormals() const {
            return mNormalOffset < mVertexStride;
        }

        bool HasUvs() const {
            return mUvOffset < mVertexStride;
        }

        bool HasColors() const {
            return mColorOffset < mVertexStride;
        }

        void Destroy() {
            mVertexBuffer.Destroy();
            mIndexBuffer.Destroy();
        }
    };

    // GPU resources for one uploaded texture. Destroy() releases the RHI
    // image + sampler (deferred deletion; the RHI leak trap fires if
    // forgotten).
    struct UploadedTexture {
        rhi::Image mImage;
        rhi::Sampler mSampler;

        void Destroy() {
            mSampler.Destroy();
            mImage.Destroy();
        }
    };

    // Uploads CPU meshes/textures to GPU resources through host-visible
    // staging buffers. Failures are recorded in moe::Error.
    class Uploader {
    public:
        bool Init(rhi::Device& device);

        // Uploads the whole mesh (all primitives concatenated; attribute layout
        // must be uniform across primitives).
        bool UploadMesh(const Mesh& mesh, UploadedMesh& out);

        // Uploads a single primitive into its own buffers. The content layer
        // uses this to preserve per-material primitive boundaries (UploadMesh
        // concatenates and would lose them).
        bool UploadMeshPrimitive(const MeshPrimitive& primitive, UploadedMesh& out);

        // Overwrites vertexData.size() bytes at the start of an uploaded
        // vertex buffer (staging write + transfer->vertex-read barrier).
        // vertexData must use the same interleaved layout as the upload.
        bool UpdateMeshVertices(const UploadedMesh& mesh,
                const uint8_t* vertexData, size_t byteCount);

        // Uploads the texture pixels into a sampled image + linear sampler
        // (sRGB format when Texture::mSrgb is set). The image ends in
        // ShaderReadOnly layout; bind it with DescriptorSet::WriteImage.
        // With Texture::mMipLevels > 1, mData must contain the whole chain
        // tightly packed (level 0 first, each level's extent halved).
        bool UploadTexture(const Texture& texture, UploadedTexture& out);

        // Uploads raw bytes into a device-local buffer (e.g. instance data).
        // The buffer is created with the given usage plus TransferDst.
        // `dstStage`/`dstAccess` describe the first shader use, so the
        // transfer->read barrier targets the right stage (e.g. pass
        // kFragmentShader for a storage buffer read by a fragment shader).
        bool UploadData(const uint8_t* data, size_t byteCount, rhi::BufferUsage usage,
                rhi::Buffer& out,
                rhi::PipelineStage dstStage = rhi::PipelineStage::kVertexShader,
                rhi::Access dstAccess = rhi::Access::kShaderRead);

        // Overwrites the start of an existing device-local buffer through a
        // staging copy. Waits for completion, so the data is ready for the
        // next recorded draw (safe to call while a render pass is open: the
        // transfer and its barrier live in their own submission).
        bool UpdateBuffer(const rhi::Buffer& dst, const void* data, size_t byteCount,
                rhi::PipelineStage dstStage = rhi::PipelineStage::kVertexInput,
                rhi::Access dstAccess = rhi::Access::kVertexAttributeRead);

    private:
        // Shared staging plumbing: a host-visible transfer-source buffer, and
        // staging->buffer copy + barrier + submit.
        bool CreateStagingBuffer(size_t size, rhi::Buffer& out);
        bool UploadBytes(const uint8_t* data, size_t byteCount, const rhi::Buffer& dst,
                bool waitForCompletion,
                rhi::PipelineStage dstStage = rhi::PipelineStage::kVertexShader,
                rhi::Access dstAccess = rhi::Access::kShaderRead);
        // Creates the vertex/index buffers and uploads packed data through a
        // single staging buffer.
        bool UploadMeshData(const std::vector<uint8_t>& vertexData,
                const std::vector<uint32_t>& indexData, uint32_t stride,
                uint32_t normalOffset, uint32_t uvOffset, uint32_t colorOffset,
                UploadedMesh& out);

        rhi::Device* mDevice{nullptr};
    };
}// namespace moe::neo
