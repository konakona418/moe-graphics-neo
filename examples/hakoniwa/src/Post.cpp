#include "Post.hpp"

#include <Core/Error.hpp>

#include <cstdio>

namespace hakoniwa {
    bool Post::Init(moe::rhi::Device& device, moe::neo::Assets& assets,
            moe::neo::Renderer& renderer) {
        moe::rhi::SamplerCreateInfo samplerInfo{};
        if (!device.CreateSampler(samplerInfo, mSampler)) {
            std::fprintf(stderr, "hakoniwa: post sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }
        mProgram = assets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/post.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/post.frag.spv");
        if (!mProgram.IsValid()) {
            std::fprintf(stderr, "hakoniwa: post shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        const moe::rhi::ShaderProgram* post = assets.GetProgram(mProgram);
        mPcResolution = renderer.GetPushConstant(*post, "mResolution");
        mPcNear = renderer.GetPushConstant(*post, "mNearPlane");
        mPcFar = renderer.GetPushConstant(*post, "mFarPlane");
        mPcToonLevels = renderer.GetPushConstant(*post, "mToonLevels");
        mPcHalftoneCell = renderer.GetPushConstant(*post, "mHalftoneCell");
        mPcHalftoneAngle = renderer.GetPushConstant(*post, "mHalftoneAngle");
        mPcInk = renderer.GetPushConstant(*post, "mInkStrength");
        mPcOutlineStrength = renderer.GetPushConstant(*post, "mOutlineStrength");
        mPcOutlineThreshold = renderer.GetPushConstant(*post, "mOutlineThreshold");
        mPcGrassOutlineSuppress = renderer.GetPushConstant(*post, "mGrassOutlineSuppress");
        if (mPcResolution < 0 || mPcNear < 0 || mPcFar < 0 || mPcToonLevels < 0
                || mPcHalftoneCell < 0 || mPcHalftoneAngle < 0 || mPcInk < 0
                || mPcOutlineStrength < 0 || mPcOutlineThreshold < 0
                || mPcGrassOutlineSuppress < 0) {
            std::fprintf(stderr, "hakoniwa: post push constant names mismatch\n");
            return false;
        }
        return true;
    }

    void Post::Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
            moe::neo::RenderTargetHandle color, moe::neo::RenderTargetHandle depth,
            moe::neo::RenderTargetHandle output, uint32_t width, uint32_t height, float nearPlane,
            float farPlane, const PostParams& params) {
        moe::neo::RenderTarget* colorRT = renderer.GetRenderTarget(color);
        moe::neo::RenderTarget* depthRT = renderer.GetRenderTarget(depth);
        const glm::vec2 resolution(static_cast<float>(width), static_cast<float>(height));
        const moe::neo::PassDesc pass{"hakoniwa post",
                moe::neo::ColorAttachment(output, moe::rhi::LoadOp::kClear), {}};
        renderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState state;
            state.mDepthTest = false;
            state.mDepthWrite = false;
            state.mCullMode = moe::rhi::CullMode::kNone;
            context.SetState(state);
            context.ClearTextureBindings();
            context.BindImage(0, *colorRT->mImage);
            context.BindImage(1, *depthRT->mDepthImage);
            context.BindSampler(2, mSampler);
            context.SetPushConstant(mPcResolution, &resolution, sizeof(glm::vec2));
            context.SetPushConstant(mPcNear, &nearPlane, sizeof(float));
            context.SetPushConstant(mPcFar, &farPlane, sizeof(float));
            context.SetPushConstant(mPcToonLevels, &params.mToonLevels, sizeof(float));
            context.SetPushConstant(mPcHalftoneCell, &params.mHalftoneCell, sizeof(float));
            context.SetPushConstant(mPcHalftoneAngle, &params.mHalftoneAngle, sizeof(float));
            context.SetPushConstant(mPcInk, &params.mInk, sizeof(float));
            context.SetPushConstant(mPcOutlineStrength, &params.mOutlineStrength, sizeof(float));
            context.SetPushConstant(mPcOutlineThreshold, &params.mOutlineThreshold, sizeof(float));
            context.SetPushConstant(
                    mPcGrassOutlineSuppress, &params.mGrassOutlineSuppress, sizeof(float));
            context.DrawFullscreen(*assets.GetProgram(mProgram));
        });
    }

    void Post::Destroy() {
        mSampler.Destroy();
    }
}// namespace hakoniwa
