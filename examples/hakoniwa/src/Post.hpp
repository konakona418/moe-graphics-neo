#pragma once

#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <RHI/Sampler.hpp>

#include <cstdint>

// Post module: tone map + toon bands + halftone + outline, applied to the whole
// composited scene (terrain + clouds) into the swapchain. The outline runs on
// the shared depth so both the ground and the clouds get ink.

namespace hakoniwa {
    struct PostParams {
        float mToonLevels{5.0f};
        float mHalftoneCell{7.0f};
        float mHalftoneAngle{0.7854f};
        float mInk{0.55f};
        float mOutlineStrength{0.85f};
        float mOutlineThreshold{0.03f};
        float mGrassOutlineSuppress{1.0f};
    };

    class Post {
    public:
        bool Init(moe::rhi::Device& device, moe::neo::Assets& assets,
                moe::neo::Renderer& renderer);
        void Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
                moe::neo::RenderTargetHandle color, moe::neo::RenderTargetHandle depth,
                moe::neo::RenderTargetHandle output, uint32_t width, uint32_t height,
                float nearPlane, float farPlane, const PostParams& params);
        void Destroy();

    private:
        moe::neo::ProgramHandle mProgram;
        moe::rhi::Sampler mSampler;
        int32_t mPcResolution{-1};
        int32_t mPcNear{-1};
        int32_t mPcFar{-1};
        int32_t mPcToonLevels{-1};
        int32_t mPcHalftoneCell{-1};
        int32_t mPcHalftoneAngle{-1};
        int32_t mPcInk{-1};
        int32_t mPcOutlineStrength{-1};
        int32_t mPcOutlineThreshold{-1};
        int32_t mPcGrassOutlineSuppress{-1};
    };
}// namespace hakoniwa
