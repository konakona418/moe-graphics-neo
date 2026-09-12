#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Image.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/Sampler.hpp>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
    constexpr uint32_t kNoiseSize = 128;
    constexpr float kSeed = 13.7f;
    constexpr glm::vec3 kBoxMin(-10.0f, -1.0f, -10.0f);
    constexpr glm::vec3 kBoxMax(10.0f, 3.5f, 10.0f);

    struct NoisePushConstants {
        uint32_t mSize;
        float mTime;
        float mSeed;
        float mPadding;
    };

    struct CloudPushConstants {
        glm::vec3 mCameraPos;
        glm::vec3 mForward;
        glm::vec3 mRight;
        glm::vec3 mUp;
        glm::vec3 mSunDir;
        float mTanHalfFov;
        float mAspect;
        float mTime;
        glm::vec3 mBoxMin;
        glm::vec3 mBoxMax;
    };

    // Volumetric clouds: a raw compute pass regenerates the 3D noise volume
    // between Renderer passes (the escape hatch), then a Renderer fullscreen
    // pass raymarches it into the swapchain.
    struct CloudsData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::ProgramHandle mCloudProgram;
        moe::rhi::Image mNoiseTex;
        moe::rhi::Sampler mSampler;
        moe::rhi::Shader mNoiseComp;
        moe::rhi::ShaderProgram mNoiseProgram;
        moe::rhi::ComputePipeline mNoisePipeline;
        moe::rhi::DescriptorSetLayout mNoiseSetLayout;
        moe::rhi::DescriptorSet mNoiseSet;
        moe::rhi::ImageLayout mNoiseLayout{moe::rhi::ImageLayout::kUndefined};
        NoisePushConstants mNoisePc{};
        CloudPushConstants mCloudPc{};
        int32_t mPcCameraPos{-1};
        int32_t mPcForward{-1};
        int32_t mPcRight{-1};
        int32_t mPcUp{-1};
        int32_t mPcSunDir{-1};
        int32_t mPcTanHalfFov{-1};
        int32_t mPcAspect{-1};
        int32_t mPcTime{-1};
        int32_t mPcBoxMin{-1};
        int32_t mPcBoxMax{-1};
        float mAngle{0.0f};
    };

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<CloudsData*>(userdata);

        moe::rhi::ImageCreateInfo noiseInfo{};
        noiseInfo.mType = moe::rhi::ImageType::k3D;
        noiseInfo.mWidth = kNoiseSize;
        noiseInfo.mHeight = kNoiseSize;
        noiseInfo.mDepth = kNoiseSize;
        noiseInfo.mFormat = moe::rhi::Format::kR8G8B8A8Unorm;
        noiseInfo.mUsage = moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled;
        if (!ctx.mDevice.CreateImage(noiseInfo, data->mNoiseTex)) {
            std::fprintf(stderr, "clouds: noise image: %s\n", moe::Error::Get().c_str());
            return false;
        }

        moe::rhi::SamplerCreateInfo samplerInfo{};
        if (!ctx.mDevice.CreateSampler(samplerInfo, data->mSampler)) {
            std::fprintf(stderr, "clouds: sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // raw compute pipeline for the noise volume (outside the Renderer)
        if (!data->mNoiseComp.Load(MOE_SOURCE_DIR "/shaders/examples/clouds/clouds_noise.comp.spv",
                    moe::rhi::ShaderStage::kCompute)
                || !data->mNoiseProgram.AddShader(data->mNoiseComp)) {
            std::fprintf(stderr, "clouds: noise shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::ComputePipelineState computeState{};
        computeState.mProgram = &data->mNoiseProgram;
        if (!ctx.mDevice.GetOrCreateComputePipeline(computeState, data->mNoisePipeline)
                || !data->mNoisePipeline.GetDescriptorSetLayout(0, data->mNoiseSetLayout)
                || !ctx.mDevice.CreateDescriptorSet(data->mNoiseSetLayout, data->mNoiseSet)
                || !data->mNoiseSet.WriteImage(0, data->mNoiseTex,
                        moe::rhi::DescriptorType::kStorageImage)) {
            std::fprintf(stderr, "clouds: noise pipeline/set: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // content layer for the raymarch program + renderer
        data->mCloudProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/clouds/clouds.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/clouds/clouds.frag.spv");
        if (!data->mCloudProgram.IsValid()) {
            std::fprintf(stderr, "clouds: cloud shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                    ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "clouds: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const moe::rhi::ShaderProgram* cloud = ctx.mAssets.GetProgram(data->mCloudProgram);
        data->mPcCameraPos = data->mRenderer.GetPushConstant(*cloud, "cameraPos");
        data->mPcForward = data->mRenderer.GetPushConstant(*cloud, "forward");
        data->mPcRight = data->mRenderer.GetPushConstant(*cloud, "right");
        data->mPcUp = data->mRenderer.GetPushConstant(*cloud, "up");
        data->mPcSunDir = data->mRenderer.GetPushConstant(*cloud, "sunDir");
        data->mPcTanHalfFov = data->mRenderer.GetPushConstant(*cloud, "tanHalfFov");
        data->mPcAspect = data->mRenderer.GetPushConstant(*cloud, "aspect");
        data->mPcTime = data->mRenderer.GetPushConstant(*cloud, "time");
        data->mPcBoxMin = data->mRenderer.GetPushConstant(*cloud, "boxMin");
        data->mPcBoxMax = data->mRenderer.GetPushConstant(*cloud, "boxMax");
        if (data->mPcCameraPos < 0 || data->mPcForward < 0 || data->mPcRight < 0
                || data->mPcUp < 0 || data->mPcSunDir < 0 || data->mPcTanHalfFov < 0
                || data->mPcAspect < 0 || data->mPcTime < 0 || data->mPcBoxMin < 0
                || data->mPcBoxMax < 0) {
            std::fprintf(stderr, "clouds: push constant names mismatch\n");
            return false;
        }

        data->mNoisePc.mSize = kNoiseSize;
        data->mNoisePc.mSeed = kSeed;
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<CloudsData*>(userdata);

        static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const float time = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - start).count();

        // auto-orbit camera around the cloud box
        data->mAngle += 0.0025f;
        const float radius = 14.0f;
        const glm::vec3 target(0.0f, 2.0f, 0.0f);
        glm::vec3 cameraPos = target
                + glm::vec3(std::cos(data->mAngle), 0.0f, std::sin(data->mAngle)) * radius;
        cameraPos.y = 8.0f; // fly above the cloud field
        const glm::vec3 forward = glm::normalize(target - cameraPos);
        const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
        const glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
        const glm::vec3 up = glm::cross(right, forward);
        const float tanHalfFov = std::tan(glm::radians(60.0f) * 0.5f);
        const float aspect = static_cast<float>(ctx.mSwapchain.GetWidth())
                / static_cast<float>(ctx.mSwapchain.GetHeight());

        data->mNoisePc.mTime = time;
        data->mCloudPc.mCameraPos = cameraPos;
        data->mCloudPc.mForward = forward;
        data->mCloudPc.mRight = right;
        data->mCloudPc.mUp = up;
        data->mCloudPc.mSunDir = glm::normalize(glm::vec3(0.55f, 0.32f, -0.65f));
        data->mCloudPc.mTanHalfFov = tanHalfFov;
        data->mCloudPc.mAspect = aspect;
        data->mCloudPc.mTime = time;
        data->mCloudPc.mBoxMin = kBoxMin;
        data->mCloudPc.mBoxMax = kBoxMax;

        const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        // compute the noise volume between passes (raw RHI is the escape
        // hatch; engine barriers keep the layout bookkeeping honest)
        {
            moe::rhi::SyncInfo sync{};
            sync.mSrcStage = data->mNoiseLayout == moe::rhi::ImageLayout::kUndefined
                    ? moe::rhi::PipelineStage::kTopOfPipe
                    : moe::rhi::PipelineStage::kFragmentShader;
            sync.mSrcAccess = data->mNoiseLayout == moe::rhi::ImageLayout::kUndefined
                    ? moe::rhi::Access::kNone
                    : moe::rhi::Access::kShaderRead;
            sync.mDstStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mDstAccess = moe::rhi::Access::kShaderWrite;
            data->mRenderer.ImageBarrier(data->mNoiseTex, data->mNoiseLayout,
                    moe::rhi::ImageLayout::kGeneral, sync);
            data->mNoiseLayout = moe::rhi::ImageLayout::kGeneral;

            cmd.BindDescriptorSet(data->mNoisePipeline, data->mNoiseSet, 0);
            cmd.SetPushConstants(data->mNoisePipeline, 0, sizeof(NoisePushConstants),
                    &data->mNoisePc);
            cmd.Dispatch(data->mNoisePipeline, kNoiseSize / 4, kNoiseSize / 4, kNoiseSize / 4);

            sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
            sync.mDstStage = moe::rhi::PipelineStage::kFragmentShader;
            sync.mDstAccess = moe::rhi::Access::kShaderRead;
            data->mRenderer.ImageBarrier(data->mNoiseTex, moe::rhi::ImageLayout::kGeneral,
                    moe::rhi::ImageLayout::kShaderReadOnly, sync);
            data->mNoiseLayout = moe::rhi::ImageLayout::kShaderReadOnly;
        }

        // raymarch straight into the swapchain
        const moe::neo::PassDesc pass{"clouds", {}, {}};
        data->mRenderer.Execute(pass, [&](moe::neo::PassContext& context) {
            context.BindImage(0, data->mNoiseTex);
            context.BindSampler(1, data->mSampler);
            context.SetPushConstant(data->mPcCameraPos, &data->mCloudPc.mCameraPos,
                    sizeof(glm::vec3));
            context.SetPushConstant(data->mPcForward, &data->mCloudPc.mForward, sizeof(glm::vec3));
            context.SetPushConstant(data->mPcRight, &data->mCloudPc.mRight, sizeof(glm::vec3));
            context.SetPushConstant(data->mPcUp, &data->mCloudPc.mUp, sizeof(glm::vec3));
            context.SetPushConstant(data->mPcSunDir, &data->mCloudPc.mSunDir, sizeof(glm::vec3));
            context.SetPushConstant(data->mPcTanHalfFov, &data->mCloudPc.mTanHalfFov,
                    sizeof(float));
            context.SetPushConstant(data->mPcAspect, &data->mCloudPc.mAspect, sizeof(float));
            context.SetPushConstant(data->mPcTime, &data->mCloudPc.mTime, sizeof(float));
            context.SetPushConstant(data->mPcBoxMin, &data->mCloudPc.mBoxMin, sizeof(glm::vec3));
            context.SetPushConstant(data->mPcBoxMax, &data->mCloudPc.mBoxMax, sizeof(glm::vec3));
            context.DrawFullscreen(*ctx.mAssets.GetProgram(data->mCloudProgram));
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<CloudsData*>(userdata);
        data->mNoiseSet.Destroy();
        data->mSampler.Destroy();
        data->mNoiseTex.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    CloudsData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("clouds demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "clouds: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
