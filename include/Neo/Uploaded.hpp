#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/Image.hpp>
#include <RHI/Sampler.hpp>

#include <cstdint>

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
}// namespace moe::neo
