#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/Sampler.hpp>

#include "Neo/Mesh.hpp"
#include "Neo/Texture.hpp"

#include <cstdint>
#include <string>

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
    // image + sampler (deferred deletion; the leak trap fires if forgotten).
    struct UploadedTexture {
        rhi::Image mImage;
        rhi::Sampler mSampler;

        void Destroy() {
            mSampler.Destroy();
            mImage.Destroy();
        }
    };

    // Uploads CPU meshes/textures to GPU resources through host-visible
    // staging buffers.
    class Uploader {
    public:
        bool Init(rhi::Device& device, std::string& error);

        // Uploads the whole mesh (all primitives concatenated; attribute layout
        // must be uniform across primitives).
        bool UploadMesh(const Mesh& mesh, UploadedMesh& out, std::string& error);

        // Overwrites vertexData.size() bytes at the start of an uploaded
        // vertex buffer (staging write + transfer->vertex-read barrier).
        // vertexData must use the same interleaved layout as the upload.
        bool UpdateMeshVertices(const UploadedMesh& mesh,
                const uint8_t* vertexData, size_t byteCount, std::string& error);

        // Uploads the texture pixels into a sampled image + linear sampler
        // (sRGB format when Texture::mSrgb is set). The image ends in
        // ShaderReadOnly layout; bind it with DescriptorSet::WriteImage.
        bool UploadTexture(const Texture& texture, UploadedTexture& out, std::string& error);

    private:
        rhi::Device* mDevice{nullptr};
    };
}// namespace moe::neo