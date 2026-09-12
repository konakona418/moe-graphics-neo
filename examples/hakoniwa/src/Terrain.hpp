#pragma once

#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/Uploader.hpp>

#include <glm/glm.hpp>

#include <cstdint>

// Terrain module: a procedural heightfield mesh rendered into the scene target
// (HDR colour + depth) for the clouds to composite over.

namespace hakoniwa {
    struct TerrainParams {
        float mSize{600.0f};
        uint32_t mResolution{256};
        float mAmplitude{20.0f};
        float mFogDensity{0.0009f};
    };

    struct TerrainFrame {
        glm::mat4 mViewProj{1.0f};
        glm::vec3 mCameraPos{0.0f};
        glm::vec3 mSunDir{0.0f, 1.0f, 0.0f};
        glm::vec3 mSunColor{1.0f};
        glm::vec3 mAmbient{0.6f, 0.62f, 0.7f};
        glm::vec3 mFogColor{0.6f, 0.7f, 0.85f};
    };

    class Terrain {
    public:
        bool Init(moe::rhi::Device& device, moe::neo::Assets& assets,
                moe::neo::Renderer& renderer, const TerrainParams& params);
        void Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
                moe::neo::RenderTargetHandle target, const TerrainFrame& frame,
                const TerrainParams& params);
        // Top-down orthographic render into the minimap scene target (the
        // minimap contour pass reads its depth).
        void RecordMinimap(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
                moe::neo::RenderTargetHandle target, const TerrainParams& params);
        void Destroy();

    private:
        moe::neo::Uploader mUploader;
        moe::neo::ProgramHandle mProgram;
        moe::neo::UploadedMesh mMesh;
        int32_t mPcViewProj{-1};
        int32_t mPcModel{-1};
        int32_t mPcCameraPos{-1};
        int32_t mPcSunDir{-1};
        int32_t mPcSunColor{-1};
        int32_t mPcAmbient{-1};
        int32_t mPcFogColor{-1};
        int32_t mPcFogDensity{-1};
    };
}// namespace hakoniwa
