#include <examples/common/App.hpp>

#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Image.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/RenderGraph.hpp>
#include <RHI/Sampler.hpp>
#include <RHI/Shader.hpp>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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

    // Compute pass: regenerates the 3D cloud noise volume.
    struct NoisePass : moe::rhi::Pass {
        moe::rhi::ComputePipeline* mPipeline{nullptr};
        moe::rhi::DescriptorSet* mSet{nullptr};
        NoisePushConstants* mPc{nullptr};

        bool Execute(moe::rhi::CommandList& cmd) override {
            cmd.SetPushConstants(*mPipeline, 0, sizeof(NoisePushConstants), mPc);
            cmd.BindDescriptorSet(*mPipeline, *mSet, 0);
            cmd.Dispatch(*mPipeline, kNoiseSize / 4, kNoiseSize / 4, kNoiseSize / 4);
            return true;
        }
    };

    // Graphics pass: raymarches the volume into the offscreen color target.
    struct CloudPass : moe::rhi::Pass {
        moe::rhi::GraphicsPipeline* mPipeline{nullptr};
        moe::rhi::DescriptorSet* mSet{nullptr};
        moe::rhi::Image* mColorTarget{nullptr};
        CloudPushConstants* mPc{nullptr};
        uint32_t mWidth{0};
        uint32_t mHeight{0};

        bool Execute(moe::rhi::CommandList& cmd) override {
            const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            cmd.BeginRendering(*mColorTarget, clear);
            cmd.BindGraphicsPipeline(*mPipeline);
            cmd.SetViewport(mWidth, mHeight);
            cmd.BindDescriptorSet(*mPipeline, *mSet, 0);
            cmd.SetPushConstants(*mPipeline, 0, sizeof(CloudPushConstants), mPc);
            cmd.Draw(3, 1, 0, 0);
            cmd.EndRendering();
            return true;
        }
    };

    struct CloudsData {
        moe::rhi::Image mNoiseTex;
        moe::rhi::Image mColorTarget;
        moe::rhi::Sampler mSampler;
        moe::rhi::Shader mNoiseComp;
        moe::rhi::Shader mVert;
        moe::rhi::Shader mFrag;
        moe::rhi::ShaderProgram mNoiseProgram;
        moe::rhi::ShaderProgram mCloudProgram;
        moe::rhi::ComputePipeline mNoisePipeline;
        moe::rhi::GraphicsPipeline mCloudPipeline;
        moe::rhi::DescriptorSetLayout mNoiseSetLayout;
        moe::rhi::DescriptorSetLayout mCloudSetLayout;
        moe::rhi::DescriptorSet mNoiseSet;
        moe::rhi::DescriptorSet mCloudSet;
        moe::rhi::RenderGraph mGraph;
        NoisePushConstants mNoisePc{};
        CloudPushConstants mCloudPc{};
        NoisePass mNoisePass;
        CloudPass mCloudPass;
    };

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<CloudsData*>(userdata);
        std::string error;

        moe::rhi::ImageCreateInfo noiseInfo{};
        noiseInfo.mType = moe::rhi::ImageType::k3D;
        noiseInfo.mWidth = kNoiseSize;
        noiseInfo.mHeight = kNoiseSize;
        noiseInfo.mDepth = kNoiseSize;
        noiseInfo.mFormat = moe::rhi::Format::kR8G8B8A8Unorm;
        noiseInfo.mUsage = moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled;
        if (!ctx.mDevice.CreateImage(noiseInfo, data->mNoiseTex)) {
            std::fprintf(stderr, "clouds: noise image: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }

        moe::rhi::ImageCreateInfo targetInfo{};
        targetInfo.mType = moe::rhi::ImageType::k2D;
        targetInfo.mWidth = ctx.mSwapchain.GetWidth();
        targetInfo.mHeight = ctx.mSwapchain.GetHeight();
        targetInfo.mDepth = 1;
        targetInfo.mFormat = ctx.mSwapchain.GetFormat();
        targetInfo.mUsage = moe::rhi::ImageUsage::kColorAttachment | moe::rhi::ImageUsage::kTransferSrc;
        if (!ctx.mDevice.CreateImage(targetInfo, data->mColorTarget)) {
            std::fprintf(stderr, "clouds: color target: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }

        moe::rhi::SamplerCreateInfo samplerInfo{};
        if (!ctx.mDevice.CreateSampler(samplerInfo, data->mSampler)) {
            std::fprintf(stderr, "clouds: sampler: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }

        if (!data->mNoiseComp.Load(MOE_SOURCE_DIR "/shaders/examples/clouds_noise.comp.spv",
                        moe::rhi::ShaderStage::kCompute)
                || !data->mVert.Load(MOE_SOURCE_DIR "/shaders/examples/clouds.vert.spv",
                        moe::rhi::ShaderStage::kVertex)
                || !data->mFrag.Load(MOE_SOURCE_DIR "/shaders/examples/clouds.frag.spv",
                        moe::rhi::ShaderStage::kFragment)) {
            std::fprintf(stderr, "clouds: shader load failed\n");
            return false;
        }
        if (!data->mNoiseProgram.AddShader(data->mNoiseComp)
                || !data->mCloudProgram.AddShader(data->mVert)
                || !data->mCloudProgram.AddShader(data->mFrag)) {
            std::fprintf(stderr, "clouds: program add failed\n");
            return false;
        }

        moe::rhi::ComputePipelineState computeState{};
        computeState.mProgram = &data->mNoiseProgram;
        if (!ctx.mDevice.GetOrCreateComputePipeline(computeState, data->mNoisePipeline)) {
            std::fprintf(stderr, "clouds: compute pipeline: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }

        moe::rhi::GraphicsPipelineState graphicsState{};
        graphicsState.mProgram = &data->mCloudProgram;
        graphicsState.mTopology = moe::rhi::PrimitiveTopology::kTriangleList;
        graphicsState.mRaster.mCullMode = moe::rhi::CullMode::kNone;
        graphicsState.mColorFormatCount = 1;
        graphicsState.mColorFormats[0] = ctx.mSwapchain.GetFormat();
        graphicsState.mBlendAttachmentCount = 1;
        if (!ctx.mDevice.GetOrCreateGraphicsPipeline(graphicsState, data->mCloudPipeline)) {
            std::fprintf(stderr, "clouds: graphics pipeline: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }

        if (!data->mNoisePipeline.GetDescriptorSetLayout(0, data->mNoiseSetLayout)
                || !data->mCloudPipeline.GetDescriptorSetLayout(0, data->mCloudSetLayout)) {
            std::fprintf(stderr, "clouds: no descriptor set layout 0\n");
            return false;
        }
        if (!ctx.mDevice.CreateDescriptorSet(data->mNoiseSetLayout, data->mNoiseSet)
                || !ctx.mDevice.CreateDescriptorSet(data->mCloudSetLayout, data->mCloudSet)) {
            std::fprintf(stderr, "clouds: descriptor set: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }        if (!data->mNoiseSet.WriteImage(0, data->mNoiseTex, moe::rhi::DescriptorType::kStorageImage)
                || !data->mCloudSet.WriteImage(0, data->mNoiseTex, moe::rhi::DescriptorType::kSampledImage)
                || !data->mCloudSet.WriteSampler(1, data->mSampler)) {
            std::fprintf(stderr, "clouds: descriptor write failed\n");
            return false;
        }

        // wire the graph: compute noise -> 3D texture, then raymarch -> color target
        data->mNoisePass.mPipeline = &data->mNoisePipeline;
        data->mNoisePass.mSet = &data->mNoiseSet;
        data->mNoisePass.mPc = &data->mNoisePc;
        data->mCloudPass.mPipeline = &data->mCloudPipeline;
        data->mCloudPass.mSet = &data->mCloudSet;
        data->mCloudPass.mColorTarget = &data->mColorTarget;
        data->mCloudPass.mPc = &data->mCloudPc;
        data->mCloudPass.mWidth = ctx.mSwapchain.GetWidth();
        data->mCloudPass.mHeight = ctx.mSwapchain.GetHeight();

        const auto noiseId = data->mGraph.RegisterImage(data->mNoiseTex);
        const auto targetId = data->mGraph.RegisterImage(data->mColorTarget);

        {
            moe::rhi::PassDesc desc{};
            desc.mName = "cloud-noise";
            desc.mPass = &data->mNoisePass;
            moe::rhi::ResourceAccess access{};
            access.mResource = noiseId;
            access.mIsWrite = true;
            access.mStage = moe::rhi::PipelineStage::kComputeShader;
            access.mAccess = moe::rhi::Access::kShaderWrite;
            access.mLayout = moe::rhi::ImageLayout::kGeneral; // storage image
            desc.mWrites.push_back(access);
            if (!data->mGraph.AddPass(desc)) {
                std::fprintf(stderr, "clouds: add noise pass failed\n");
                return false;
            }
        }
        {
            moe::rhi::PassDesc desc{};
            desc.mName = "cloud-raymarch";
            desc.mPass = &data->mCloudPass;
            moe::rhi::ResourceAccess access{};
            access.mResource = noiseId;
            access.mIsWrite = false;
            access.mStage = moe::rhi::PipelineStage::kFragmentShader;
            access.mAccess = moe::rhi::Access::kShaderRead;
            access.mLayout = moe::rhi::ImageLayout::kShaderReadOnly;
            desc.mReads.push_back(access);
            access.mResource = targetId;
            access.mIsWrite = true;
            access.mStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
            access.mAccess = moe::rhi::Access::kColorAttachmentWrite;
            access.mLayout = moe::rhi::ImageLayout::kColorAttachment;
            desc.mWrites.push_back(access);
            if (!data->mGraph.AddPass(desc)) {
                std::fprintf(stderr, "clouds: add raymarch pass failed\n");
                return false;
            }
        }
        std::string graphError;
        if (!data->mGraph.Compile(graphError)) {
            std::fprintf(stderr, "clouds: graph compile: %s\n", graphError.c_str());
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
        static float angle = 0.0f;
        angle += 0.0025f;
        const float radius = 14.0f;
        const glm::vec3 target(0.0f, 2.0f, 0.0f);
        glm::vec3 cameraPos = target
                + glm::vec3(std::cos(angle), 0.0f, std::sin(angle)) * radius;
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

        // graph: compute noise (general) -> raymarch (color target, shader-read noise)
        if (!data->mGraph.Execute(cmd)) {
            std::fprintf(stderr, "clouds: graph execute failed\n");
            return;
        }

        // copy the offscreen color target into the swapchain image (currently in
        // PresentSrc layout from App's EndRendering)
        moe::rhi::Image swapImage;
        if (!ctx.mSwapchain.GetCurrentImage(swapImage)) {
            return;
        }

        moe::rhi::SyncInfo sync{};
        // color target: color attachment -> transfer src
        sync.mSrcStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
        sync.mSrcAccess = moe::rhi::Access::kColorAttachmentWrite;
        sync.mDstStage = moe::rhi::PipelineStage::kTransfer;
        sync.mDstAccess = moe::rhi::Access::kTransferRead;
        cmd.ImageBarrier(data->mColorTarget, moe::rhi::ImageLayout::kColorAttachment,
                moe::rhi::ImageLayout::kTransferSrc, sync);
        // swapchain: present src -> transfer dst
        sync.mSrcStage = moe::rhi::PipelineStage::kBottomOfPipe;
        sync.mSrcAccess = moe::rhi::Access::kNone;
        sync.mDstStage = moe::rhi::PipelineStage::kTransfer;
        sync.mDstAccess = moe::rhi::Access::kTransferWrite;
        cmd.ImageBarrier(swapImage, moe::rhi::ImageLayout::kPresentSrc,
                moe::rhi::ImageLayout::kTransferDst, sync);
        cmd.CopyImage(data->mColorTarget, moe::rhi::ImageLayout::kTransferSrc,
                swapImage, moe::rhi::ImageLayout::kTransferDst);
        // swapchain: transfer dst -> present src (for Present)
        sync.mSrcStage = moe::rhi::PipelineStage::kTransfer;
        sync.mSrcAccess = moe::rhi::Access::kTransferWrite;
        sync.mDstStage = moe::rhi::PipelineStage::kBottomOfPipe;
        sync.mDstAccess = moe::rhi::Access::kNone;
        cmd.ImageBarrier(swapImage, moe::rhi::ImageLayout::kTransferDst,
                moe::rhi::ImageLayout::kPresentSrc, sync);

        swapImage.Destroy(); // borrowed wrapper: only drops the wrapper
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<CloudsData*>(userdata);
        data->mNoiseSet.Destroy();
        data->mCloudSet.Destroy();
        data->mSampler.Destroy();
        data->mNoiseTex.Destroy();
        data->mColorTarget.Destroy();
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
    std::string error;
    if (!app.Run("clouds demo", 1280, 720, callbacks, error)) {
        std::fprintf(stderr, "clouds: app: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
