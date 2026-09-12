#pragma once

#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <RHI/Sampler.hpp>

#include <cstdint>

// Minimap module: reads the top-down terrain depth (rendered by
// Terrain::RecordMinimap) and writes an elevation tint + contour lines, which
// the HUD shows as an image.

namespace hakoniwa {
    class Minimap {
    public:
        bool Init(moe::rhi::Device& device, moe::neo::Assets& assets,
                moe::neo::Renderer& renderer);
        void Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
                moe::neo::RenderTargetHandle sceneTarget, moe::neo::RenderTargetHandle target,
                float terrainAmplitude);
        void Destroy();

        const moe::rhi::Sampler& Sampler() const { return mSampler; }

    private:
        moe::neo::ProgramHandle mProgram;
        moe::rhi::Sampler mSampler;
        int32_t mPcCameraY{-1};
        int32_t mPcNear{-1};
        int32_t mPcFar{-1};
        int32_t mPcInterval{-1};
        int32_t mPcMinHeight{-1};
        int32_t mPcMaxHeight{-1};
    };
}// namespace hakoniwa
