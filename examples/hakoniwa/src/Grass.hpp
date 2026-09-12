#pragma once

#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/Uploader.hpp>
#include <RHI/CommandList.hpp>

#include <glm/glm.hpp>

#include <cstdint>

// Grass module: a clipmap of procedural blades over the terrain, with the LOD
// computed entirely in the vertex shader (concentric rings, see grass.slang).
// One instanced draw per ring; no per-frame CPU work and no instance buffer.

namespace hakoniwa {
    struct GrassParams {
        float mBaseTileSize{2.0f};
        int32_t mTilesPerSide{32};
        int32_t mBladesPerTile{128};
        int32_t mRingCount{4};
        float mWindStrength{1.0f};

        // Player displacement field.
        float mFieldSize{24.0f};
        float mFieldRadius{1.6f};
        float mFieldStrength{0.6f};
        float mFieldDecay{0.88f};
        float mFieldDisplace{1.0f};
    };

    struct GrassFrame {
        glm::mat4 mViewProj{1.0f};
        glm::vec3 mCameraPos{0.0f};
        glm::vec3 mSunDir{0.0f, 1.0f, 0.0f};
        glm::vec3 mSunColor{1.0f};
        glm::vec3 mAmbient{0.6f, 0.62f, 0.7f};
        glm::vec3 mFogColor{0.6f, 0.7f, 0.85f};
        glm::vec3 mPlayerPos{0.0f};
        float mFogDensity{0.0f};
        float mTerrainAmplitude{20.0f};
        float mTerrainHalfSize{300.0f};
        float mTime{0.0f};
        float mDeltaTime{0.016f};
    };

    class Grass {
    public:
        bool Init(moe::rhi::Device& device, moe::neo::Assets& assets,
                moe::neo::Renderer& renderer);
        void Record(moe::rhi::CommandList& cmd, moe::neo::Assets& assets,
                moe::neo::Renderer& renderer, moe::neo::RenderTargetHandle target,
                const GrassFrame& frame, const GrassParams& params);
        void Destroy();

    private:
        moe::neo::Uploader mUploader;
        moe::neo::ProgramHandle mProgram;
        moe::neo::UploadedMesh mMesh;
        int32_t mPcViewProj{-1};
        int32_t mPcCameraPos{-1};
        int32_t mPcSunDir{-1};
        int32_t mPcSunColor{-1};
        int32_t mPcAmbient{-1};
        int32_t mPcFogColor{-1};
        int32_t mPcFogDensity{-1};
        int32_t mPcAmplitude{-1};
        int32_t mPcTileSize{-1};
        int32_t mPcTilesPerSide{-1};
        int32_t mPcBladesPerTile{-1};
        int32_t mPcInnerTiles{-1};
        int32_t mPcOuterTiles{-1};
        int32_t mPcWindStrength{-1};
        int32_t mPcTime{-1};
        int32_t mPcTerrainHalfSize{-1};
        int32_t mPcFieldOrigin{-1};
        int32_t mPcFieldSize{-1};
        int32_t mPcFieldDisplace{-1};

        // Player displacement field (compute-written storage image, sampled by
        // the grass vertex shader).
        moe::rhi::Image mField;
        moe::rhi::Sampler mFieldSampler;
        moe::rhi::Shader mDisplaceComp;
        moe::rhi::ShaderProgram mDisplaceProgram;
        moe::rhi::ComputePipeline mDisplacePipeline;
        moe::rhi::DescriptorSet mDisplaceSet;
        moe::rhi::ImageLayout mFieldLayout{moe::rhi::ImageLayout::kUndefined};
    };
}// namespace hakoniwa
