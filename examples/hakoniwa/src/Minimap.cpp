#include "Minimap.hpp"

#include <Core/Error.hpp>

#include <cstdio>

namespace hakoniwa {
    bool Minimap::Init(moe::rhi::Device& device, moe::neo::Assets& assets,
            moe::neo::Renderer& renderer) {
        mProgram = assets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/minimap.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/minimap.frag.spv");
        if (!mProgram.IsValid()) {
            std::fprintf(stderr, "hakoniwa: minimap shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        const moe::rhi::ShaderProgram* minimap = assets.GetProgram(mProgram);
        mPcCameraY = renderer.GetPushConstant(*minimap, "mCameraY");
        mPcNear = renderer.GetPushConstant(*minimap, "mNearPlane");
        mPcFar = renderer.GetPushConstant(*minimap, "mFarPlane");
        mPcInterval = renderer.GetPushConstant(*minimap, "mInterval");
        mPcMinHeight = renderer.GetPushConstant(*minimap, "mMinHeight");
        mPcMaxHeight = renderer.GetPushConstant(*minimap, "mMaxHeight");
        if (mPcCameraY < 0 || mPcNear < 0 || mPcFar < 0 || mPcInterval < 0 || mPcMinHeight < 0
                || mPcMaxHeight < 0) {
            std::fprintf(stderr, "hakoniwa: minimap push constant names mismatch\n");
            return false;
        }
        moe::rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mAddressModeU = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeV = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeW = moe::rhi::AddressMode::kClampToEdge;
        if (!device.CreateSampler(samplerInfo, mSampler)) {
            std::fprintf(stderr, "hakoniwa: minimap sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    void Minimap::Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
            moe::neo::RenderTargetHandle sceneTarget, moe::neo::RenderTargetHandle target,
            float terrainAmplitude) {
        moe::neo::RenderTarget* scene = renderer.GetRenderTarget(sceneTarget);
        if (scene == nullptr) {
            return;
        }
        const float cameraY = 900.0f;
        const float nearPlane = 0.1f;
        const float farPlane = 2000.0f;
        const float interval = 4.0f;
        const float minHeight = -terrainAmplitude;
        const float maxHeight = terrainAmplitude;
        const moe::neo::PassDesc pass{"hakoniwa minimap contour",
                moe::neo::ColorAttachment(target, moe::rhi::LoadOp::kClear), {}};
        renderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState state;
            state.mDepthTest = false;
            state.mDepthWrite = false;
            state.mCullMode = moe::rhi::CullMode::kNone;
            context.SetState(state);
            context.ClearTextureBindings();
            context.BindImage(0, *scene->mDepthImage);
            context.BindSampler(1, mSampler);
            context.SetPushConstant(mPcCameraY, &cameraY, sizeof(float));
            context.SetPushConstant(mPcNear, &nearPlane, sizeof(float));
            context.SetPushConstant(mPcFar, &farPlane, sizeof(float));
            context.SetPushConstant(mPcInterval, &interval, sizeof(float));
            context.SetPushConstant(mPcMinHeight, &minHeight, sizeof(float));
            context.SetPushConstant(mPcMaxHeight, &maxHeight, sizeof(float));
            context.DrawFullscreen(*assets.GetProgram(mProgram));
        });
    }

    void Minimap::Destroy() {
        mSampler.Destroy();
    }
}// namespace hakoniwa
