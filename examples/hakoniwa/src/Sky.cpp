#include "Sky.hpp"

#include <Core/Error.hpp>

#include <cstdio>

namespace hakoniwa {
    bool Sky::Init(moe::neo::Assets& assets, moe::neo::Renderer& renderer) {
        mProgram = assets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/sky.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/sky.frag.spv");
        if (!mProgram.IsValid()) {
            std::fprintf(stderr, "hakoniwa: sky shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        const moe::rhi::ShaderProgram* sky = assets.GetProgram(mProgram);
        mPcForward = renderer.GetPushConstant(*sky, "mForward");
        mPcRight = renderer.GetPushConstant(*sky, "mRight");
        mPcUp = renderer.GetPushConstant(*sky, "mUp");
        mPcTanHalfFov = renderer.GetPushConstant(*sky, "mTanHalfFov");
        mPcAspect = renderer.GetPushConstant(*sky, "mAspect");
        mPcSunDir = renderer.GetPushConstant(*sky, "mSunDir");
        mPcSunColor = renderer.GetPushConstant(*sky, "mSunColor");
        mPcHorizonColor = renderer.GetPushConstant(*sky, "mHorizonColor");
        mPcZenithColor = renderer.GetPushConstant(*sky, "mZenithColor");
        if (mPcForward < 0 || mPcRight < 0 || mPcUp < 0 || mPcTanHalfFov < 0 || mPcAspect < 0
                || mPcSunDir < 0 || mPcSunColor < 0 || mPcHorizonColor < 0
                || mPcZenithColor < 0) {
            std::fprintf(stderr, "hakoniwa: sky push constant names mismatch\n");
            return false;
        }
        return true;
    }

    void Sky::Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
            moe::neo::RenderTargetHandle target, const SkyFrame& frame) {
        // Clears the scene target (colour + depth); terrain/grass then load it.
        const moe::neo::PassDesc pass{"hakoniwa sky",
                moe::neo::ColorAttachment(target, moe::rhi::LoadOp::kClear),
                moe::neo::DepthAttachment(target, moe::rhi::LoadOp::kClear)};
        renderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState state;
            state.mDepthTest = false;
            state.mDepthWrite = false;
            state.mCullMode = moe::rhi::CullMode::kNone;
            context.SetState(state);
            context.ClearTextureBindings();
            context.SetPushConstant(mPcForward, &frame.mForward, sizeof(glm::vec3));
            context.SetPushConstant(mPcRight, &frame.mRight, sizeof(glm::vec3));
            context.SetPushConstant(mPcUp, &frame.mUp, sizeof(glm::vec3));
            context.SetPushConstant(mPcTanHalfFov, &frame.mTanHalfFov, sizeof(float));
            context.SetPushConstant(mPcAspect, &frame.mAspect, sizeof(float));
            context.SetPushConstant(mPcSunDir, &frame.mSunDir, sizeof(glm::vec3));
            context.SetPushConstant(mPcSunColor, &frame.mSunColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcHorizonColor, &frame.mHorizonColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcZenithColor, &frame.mZenithColor, sizeof(glm::vec3));
            context.DrawFullscreen(*assets.GetProgram(mProgram));
        });
    }

    void Sky::Destroy() {}
}// namespace hakoniwa
