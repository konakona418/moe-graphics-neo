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

#include <imgui.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
    constexpr uint32_t kVolumeW = 160;
    constexpr uint32_t kVolumeH = 80;
    constexpr uint32_t kVolumeD = 160;
    constexpr uint32_t kNoiseSize = 128;
    constexpr uint32_t kGridW = 128;
    constexpr uint32_t kGridH = 64;
    constexpr uint32_t kGridD = 128;

    const glm::vec3 kBoxMin(-400.0f, -200.0f, -400.0f);
    const glm::vec3 kBoxMax(400.0f, 200.0f, 400.0f);

    struct VolumePushConstants {
        glm::vec3 mBoxMin;
        glm::vec3 mBoxMax;
        glm::vec3 mGridSize;
        float mTime;
        float mSeed;
        float mPad0;
        float mPad1;
    };

    struct NoisePushConstants {
        uint32_t mSize;
        float mSeed;
        float mPad0;
        float mPad1;
    };

    struct LightGridPushConstants {
        glm::vec3 mBoxMin;
        glm::vec3 mBoxMax;
        glm::vec3 mGridSize;
        glm::vec3 mSunDir;
        float mPad0;
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
        float mFarClip;
        float mTransmittanceLimit;
        float mTilingFreq;
        float mDensityScale;
        float mTurbidity;
    };

    // Nubis3-style clouds. Three one-time compute passes author the NVDF
    // volume, the detail-noise volume and the light grid; a Renderer fullscreen
    // pass sphere-traces them into the swapchain.
    struct NubisData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::ProgramHandle mCloudProgram;
        moe::neo::ProgramHandle mCompositeProgram;
        moe::neo::RenderTargetHandle mHdrTarget;

        moe::rhi::Image mVolume;
        moe::rhi::Image mNoise;
        moe::rhi::Image mLightGrid;
        moe::rhi::Sampler mSampler;

        moe::rhi::Shader mVolumeComp;
        moe::rhi::Shader mNoiseComp;
        moe::rhi::Shader mLightGridComp;
        moe::rhi::ShaderProgram mVolumeProgram;
        moe::rhi::ShaderProgram mNoiseProgram;
        moe::rhi::ShaderProgram mLightGridProgram;
        moe::rhi::ComputePipeline mVolumePipeline;
        moe::rhi::ComputePipeline mNoisePipeline;
        moe::rhi::ComputePipeline mLightGridPipeline;
        moe::rhi::DescriptorSetLayout mVolumeSetLayout;
        moe::rhi::DescriptorSetLayout mNoiseSetLayout;
        moe::rhi::DescriptorSetLayout mLightGridSetLayout;
        moe::rhi::DescriptorSet mVolumeSet;
        moe::rhi::DescriptorSet mNoiseSet;
        moe::rhi::DescriptorSet mLightGridSet;

        moe::rhi::ImageLayout mVolumeLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mNoiseLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mLightGridLayout{moe::rhi::ImageLayout::kUndefined};

        VolumePushConstants mVolumePc{};
        NoisePushConstants mNoisePc{};
        LightGridPushConstants mLightGridPc{};
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
        int32_t mPcFarClip{-1};
        int32_t mPcTransmittanceLimit{-1};
        int32_t mPcTilingFreq{-1};
        int32_t mPcDensityScale{-1};
        int32_t mPcTurbidity{-1};
        int32_t mPcSunScreen{-1};
        int32_t mPcSunElevation{-1};
        int32_t mPcGodrayEnabled{-1};
        int32_t mPcGodrayExposure{-1};
        int32_t mPcExposure{-1};
        int32_t mPcWhitePoint{-1};

        glm::vec2 mSunScreen{0.0f, 0.0f};
        float mDensityScale{5.0f};
        float mTilingFreq{1.0f};
        float mFarClip{4000.0f};
        float mTransmittanceLimit{0.01f};
        float mTurbidity{2.2f};
        float mGodrayExposure{0.09f};
        float mExposure{0.7f};
        float mWhitePoint{1.0f};
        bool mGodrayEnabled{true};
        float mSunElevationDeg{55.0f};
        float mSunAzimuthDeg{30.0f};
        bool mGenerated{false};
        bool mRegenerate{false};
        float mOrbitAngle{0.0f};
        float mOrbitRadius{900.0f};
        float mOrbitHeight{190.0f};
        bool mAutoOrbit{true};
    };

    bool CreateVolume(moe::rhi::Device& device, uint32_t w, uint32_t h, uint32_t d,
            moe::rhi::Format format, moe::rhi::Image& out) {
        moe::rhi::ImageCreateInfo info{};
        info.mType = moe::rhi::ImageType::k3D;
        info.mWidth = w;
        info.mHeight = h;
        info.mDepth = d;
        info.mFormat = format;
        info.mUsage = moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled;
        if (!device.CreateImage(info, out)) {
            std::fprintf(stderr, "nubis: volume image: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    bool SetupCompute(moe::rhi::Device& device, const char* path,
            moe::rhi::Shader& shader, moe::rhi::ShaderProgram& program,
            moe::rhi::ComputePipeline& pipeline, moe::rhi::DescriptorSetLayout& layout,
            moe::rhi::DescriptorSet& set) {
        if (!shader.Load(path, moe::rhi::ShaderStage::kCompute) || !program.AddShader(shader)) {
            std::fprintf(stderr, "nubis: compute shader %s: %s\n", path,
                    moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::ComputePipelineState state{};
        state.mProgram = &program;
        if (!device.GetOrCreateComputePipeline(state, pipeline)
                || !pipeline.GetDescriptorSetLayout(0, layout)
                || !device.CreateDescriptorSet(layout, set)) {
            std::fprintf(stderr, "nubis: compute pipeline %s: %s\n", path,
                    moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    glm::vec3 MakeSunDirection(float elevationDeg, float azimuthDeg) {
        const float e = glm::radians(elevationDeg);
        const float a = glm::radians(azimuthDeg);
        return glm::normalize(
                glm::vec3(glm::cos(e) * glm::sin(a), glm::sin(e), glm::cos(e) * glm::cos(a)));
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<NubisData*>(userdata);

        if (!CreateVolume(ctx.mDevice, kVolumeW, kVolumeH, kVolumeD,
                    moe::rhi::Format::kR32G32B32A32Float, data->mVolume)
                || !CreateVolume(ctx.mDevice, kNoiseSize, kNoiseSize, kNoiseSize,
                        moe::rhi::Format::kR8G8B8A8Unorm, data->mNoise)
                || !CreateVolume(ctx.mDevice, kGridW, kGridH, kGridD,
                        moe::rhi::Format::kR32G32B32A32Float, data->mLightGrid)) {
            return false;
        }

        moe::rhi::SamplerCreateInfo samplerInfo{};
        if (!ctx.mDevice.CreateSampler(samplerInfo, data->mSampler)) {
            std::fprintf(stderr, "nubis: sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!SetupCompute(ctx.mDevice,
                    MOE_SOURCE_DIR "/shaders/examples/nubis_volume.comp.spv",
                    data->mVolumeComp, data->mVolumeProgram, data->mVolumePipeline,
                    data->mVolumeSetLayout, data->mVolumeSet)
                || !SetupCompute(ctx.mDevice,
                        MOE_SOURCE_DIR "/shaders/examples/nubis_noise.comp.spv",
                        data->mNoiseComp, data->mNoiseProgram, data->mNoisePipeline,
                        data->mNoiseSetLayout, data->mNoiseSet)
                || !SetupCompute(ctx.mDevice,
                        MOE_SOURCE_DIR "/shaders/examples/nubis_lightgrid.comp.spv",
                        data->mLightGridComp, data->mLightGridProgram, data->mLightGridPipeline,
                        data->mLightGridSetLayout, data->mLightGridSet)) {
            return false;
        }

        if (!data->mVolumeSet.WriteImage(0, data->mVolume,
                    moe::rhi::DescriptorType::kStorageImage)
                || !data->mNoiseSet.WriteImage(0, data->mNoise,
                        moe::rhi::DescriptorType::kStorageImage)
                || !data->mLightGridSet.WriteImage(0, data->mLightGrid,
                        moe::rhi::DescriptorType::kStorageImage)
                || !data->mLightGridSet.WriteImage(1, data->mVolume,
                        moe::rhi::DescriptorType::kSampledImage)
                || !data->mLightGridSet.WriteSampler(2, data->mSampler)) {
            std::fprintf(stderr, "nubis: descriptor writes: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mCloudProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/nubis.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/nubis.frag.spv");
        if (!data->mCloudProgram.IsValid()) {
            std::fprintf(stderr, "nubis: cloud shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                    ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "nubis: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mHdrTarget = data->mRenderer.CreateRenderTarget(
                ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(),
                moe::rhi::Format::kR16G16B16A16Float, true, 1);
        if (!data->mHdrTarget.IsValid()) {
            std::fprintf(stderr, "nubis: hdr target: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mCompositeProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/nubis_composite.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/nubis_composite.frag.spv");
        if (!data->mCompositeProgram.IsValid()) {
            std::fprintf(stderr, "nubis: composite shader: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const moe::rhi::ShaderProgram* cloud = ctx.mAssets.GetProgram(data->mCloudProgram);
        data->mPcCameraPos = data->mRenderer.GetPushConstant(*cloud, "mCameraPos");
        data->mPcForward = data->mRenderer.GetPushConstant(*cloud, "mForward");
        data->mPcRight = data->mRenderer.GetPushConstant(*cloud, "mRight");
        data->mPcUp = data->mRenderer.GetPushConstant(*cloud, "mUp");
        data->mPcSunDir = data->mRenderer.GetPushConstant(*cloud, "mSunDir");
        data->mPcTanHalfFov = data->mRenderer.GetPushConstant(*cloud, "mTanHalfFov");
        data->mPcAspect = data->mRenderer.GetPushConstant(*cloud, "mAspect");
        data->mPcTime = data->mRenderer.GetPushConstant(*cloud, "mTime");
        data->mPcBoxMin = data->mRenderer.GetPushConstant(*cloud, "mBoxMin");
        data->mPcBoxMax = data->mRenderer.GetPushConstant(*cloud, "mBoxMax");
        data->mPcFarClip = data->mRenderer.GetPushConstant(*cloud, "mFarClip");
        data->mPcTransmittanceLimit =
                data->mRenderer.GetPushConstant(*cloud, "mTransmittanceLimit");
        data->mPcTilingFreq = data->mRenderer.GetPushConstant(*cloud, "mTilingFreq");
        data->mPcDensityScale = data->mRenderer.GetPushConstant(*cloud, "mDensityScale");
        data->mPcTurbidity = data->mRenderer.GetPushConstant(*cloud, "mTurbidity");
        if (data->mPcCameraPos < 0 || data->mPcForward < 0 || data->mPcRight < 0
                || data->mPcUp < 0 || data->mPcSunDir < 0 || data->mPcTanHalfFov < 0
                || data->mPcAspect < 0 || data->mPcTime < 0 || data->mPcBoxMin < 0
                || data->mPcBoxMax < 0 || data->mPcFarClip < 0
                || data->mPcTransmittanceLimit < 0 || data->mPcTilingFreq < 0
                || data->mPcDensityScale < 0 || data->mPcTurbidity < 0) {
            std::fprintf(stderr, "nubis: push constant names mismatch\n");
            return false;
        }

        const moe::rhi::ShaderProgram* composite =
                ctx.mAssets.GetProgram(data->mCompositeProgram);
        data->mPcSunScreen = data->mRenderer.GetPushConstant(*composite, "mSunScreen");
        data->mPcSunElevation = data->mRenderer.GetPushConstant(*composite, "mSunElevation");
        data->mPcGodrayEnabled = data->mRenderer.GetPushConstant(*composite, "mGodrayEnabled");
        data->mPcGodrayExposure = data->mRenderer.GetPushConstant(*composite, "mGodrayExposure");
        data->mPcExposure = data->mRenderer.GetPushConstant(*composite, "mExposure");
        data->mPcWhitePoint = data->mRenderer.GetPushConstant(*composite, "mWhitePoint");
        if (data->mPcSunScreen < 0 || data->mPcSunElevation < 0 || data->mPcGodrayEnabled < 0
                || data->mPcGodrayExposure < 0 || data->mPcExposure < 0
                || data->mPcWhitePoint < 0) {
            std::fprintf(stderr, "nubis: composite push constant names mismatch\n");
            return false;
        }

        data->mVolumePc.mBoxMin = kBoxMin;
        data->mVolumePc.mBoxMax = kBoxMax;
        data->mVolumePc.mGridSize = glm::vec3(float(kVolumeW), float(kVolumeH), float(kVolumeD));
        data->mVolumePc.mSeed = 3.7f;
        data->mNoisePc.mSize = kNoiseSize;
        data->mNoisePc.mSeed = 1.3f;
        data->mLightGridPc.mBoxMin = kBoxMin;
        data->mLightGridPc.mBoxMax = kBoxMax;
        data->mLightGridPc.mGridSize = glm::vec3(float(kGridW), float(kGridH), float(kGridD));
        data->mLightGridPc.mSunDir =
                MakeSunDirection(data->mSunElevationDeg, data->mSunAzimuthDeg);
        data->mCloudPc.mFarClip = data->mFarClip;
        data->mCloudPc.mTransmittanceLimit = data->mTransmittanceLimit;
        data->mCloudPc.mTilingFreq = data->mTilingFreq;
        data->mCloudPc.mDensityScale = data->mDensityScale;
        data->mCloudPc.mTurbidity = data->mTurbidity;
        return true;
    }

    void GenerateVolumes(NubisData* data, examples::AppContext& ctx,
            moe::rhi::CommandList& cmd) {
        moe::rhi::SyncInfo sync{};
        auto toGeneral = [&](moe::rhi::Image& image, moe::rhi::ImageLayout& layout) {
            sync.mSrcStage = layout == moe::rhi::ImageLayout::kUndefined
                    ? moe::rhi::PipelineStage::kTopOfPipe
                    : moe::rhi::PipelineStage::kComputeShader;
            sync.mSrcAccess = layout == moe::rhi::ImageLayout::kUndefined
                    ? moe::rhi::Access::kNone
                    : moe::rhi::Access::kShaderRead;
            sync.mDstStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mDstAccess = moe::rhi::Access::kShaderWrite;
            data->mRenderer.ImageBarrier(image, layout, moe::rhi::ImageLayout::kGeneral, sync);
            layout = moe::rhi::ImageLayout::kGeneral;
        };
        auto toRead = [&](moe::rhi::Image& image, moe::rhi::ImageLayout& layout,
                              moe::rhi::PipelineStage dstStage) {
            sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
            sync.mDstStage = dstStage;
            sync.mDstAccess = moe::rhi::Access::kShaderRead;
            data->mRenderer.ImageBarrier(image, layout, moe::rhi::ImageLayout::kShaderReadOnly,
                    sync);
            layout = moe::rhi::ImageLayout::kShaderReadOnly;
        };

        data->mVolumePc.mTime = 0.0f;
        toGeneral(data->mVolume, data->mVolumeLayout);
        cmd.BindDescriptorSet(data->mVolumePipeline, data->mVolumeSet, 0);
        cmd.SetPushConstants(data->mVolumePipeline, 0, sizeof(VolumePushConstants),
                &data->mVolumePc);
        cmd.Dispatch(data->mVolumePipeline, kVolumeW / 4, kVolumeH / 4, kVolumeD / 4);
        // the light grid reads the NVDF, so publish it to compute first
        toRead(data->mVolume, data->mVolumeLayout, moe::rhi::PipelineStage::kComputeShader);

        toGeneral(data->mNoise, data->mNoiseLayout);
        cmd.BindDescriptorSet(data->mNoisePipeline, data->mNoiseSet, 0);
        cmd.SetPushConstants(data->mNoisePipeline, 0, sizeof(NoisePushConstants), &data->mNoisePc);
        cmd.Dispatch(data->mNoisePipeline, kNoiseSize / 4, kNoiseSize / 4, kNoiseSize / 4);
        toRead(data->mNoise, data->mNoiseLayout, moe::rhi::PipelineStage::kFragmentShader);

        toGeneral(data->mLightGrid, data->mLightGridLayout);
        cmd.BindDescriptorSet(data->mLightGridPipeline, data->mLightGridSet, 0);
        cmd.SetPushConstants(data->mLightGridPipeline, 0, sizeof(LightGridPushConstants),
                &data->mLightGridPc);
        cmd.Dispatch(data->mLightGridPipeline, kGridW / 8, kGridH / 8, kGridD);
        toRead(data->mLightGrid, data->mLightGridLayout, moe::rhi::PipelineStage::kFragmentShader);

        // the raymarch samples the NVDF too; publish the earlier write to fragment
        toRead(data->mVolume, data->mVolumeLayout, moe::rhi::PipelineStage::kFragmentShader);
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<NubisData*>(userdata);

        static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const float time = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - start).count();

        // camera control: left-drag orbits, wheel zooms, space toggles auto-orbit
        const ImGuiIO& io = ImGui::GetIO();
        const moe::neo::MouseState& mouse = ctx.mInput.GetMouse();
        if (!io.WantCaptureMouse) {
            if (mouse.mButtonDown[0]) {
                data->mAutoOrbit = false;
                data->mOrbitAngle += mouse.mDeltaX * 0.005f;
                data->mOrbitHeight = glm::clamp(
                        data->mOrbitHeight + mouse.mDeltaY * 1.5f, -300.0f, 600.0f);
            }
            if (mouse.mScrollY != 0.0f) {
                data->mOrbitRadius = glm::clamp(
                        data->mOrbitRadius * std::exp(-mouse.mScrollY * 0.1f), 300.0f, 2000.0f);
            }
        }
        if (!io.WantCaptureKeyboard
                && ctx.mInput.IsKeyJustPressed(
                        static_cast<int32_t>(moe::neo::KeyCode::kSpace))) {
            data->mAutoOrbit = !data->mAutoOrbit;
        }

        if (data->mAutoOrbit) {
            data->mOrbitAngle += 0.0015f;
        }
        const glm::vec3 target(0.0f, 20.0f, 0.0f);
        glm::vec3 cameraPos = target
                + glm::vec3(std::cos(data->mOrbitAngle), 0.0f, std::sin(data->mOrbitAngle))
                        * data->mOrbitRadius;
        cameraPos.y = data->mOrbitHeight;
        const glm::vec3 forward = glm::normalize(target - cameraPos);
        const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
        const glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
        const glm::vec3 up = glm::cross(right, forward);
        const float tanHalfFov = std::tan(glm::radians(60.0f) * 0.5f);
        const float aspect = static_cast<float>(ctx.mSwapchain.GetWidth())
                / static_cast<float>(ctx.mSwapchain.GetHeight());

        data->mCloudPc.mCameraPos = cameraPos;
        data->mCloudPc.mForward = forward;
        data->mCloudPc.mRight = right;
        data->mCloudPc.mUp = up;
        data->mCloudPc.mSunDir =
                MakeSunDirection(data->mSunElevationDeg, data->mSunAzimuthDeg);
        data->mCloudPc.mTanHalfFov = tanHalfFov;
        data->mCloudPc.mAspect = aspect;
        data->mCloudPc.mTime = time;
        data->mCloudPc.mBoxMin = kBoxMin;
        data->mCloudPc.mBoxMax = kBoxMax;
        data->mCloudPc.mFarClip = data->mFarClip;
        data->mCloudPc.mTransmittanceLimit = data->mTransmittanceLimit;
        data->mCloudPc.mTilingFreq = data->mTilingFreq;
        data->mCloudPc.mDensityScale = data->mDensityScale;
        data->mCloudPc.mTurbidity = data->mTurbidity;

        // project the sun direction into screen uv for the god-ray pass. Clamp
        // forward so a sun behind the camera still yields a consistent edge
        // direction instead of collapsing to the screen centre.
        const float sunForward = glm::max(glm::dot(data->mCloudPc.mSunDir, forward), 0.001f);
        const float sx = glm::dot(data->mCloudPc.mSunDir, right)
                / (sunForward * tanHalfFov * aspect);
        const float sy = glm::dot(data->mCloudPc.mSunDir, up) / (sunForward * tanHalfFov);
        data->mSunScreen = glm::clamp(
                glm::vec2(0.5f + 0.5f * sx, 0.5f - 0.5f * sy),
                glm::vec2(-0.5f), glm::vec2(1.5f));

        const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        if (!data->mGenerated || data->mRegenerate) {
            data->mLightGridPc.mSunDir = data->mCloudPc.mSunDir;
            GenerateVolumes(data, ctx, cmd);
            data->mGenerated = true;
            data->mRegenerate = false;
        }

        const moe::neo::PassDesc raymarchPass{"nubis raymarch",
                moe::neo::ColorAttachment(data->mHdrTarget), {}};
        data->mRenderer.Execute(raymarchPass, [&](moe::neo::PassContext& context) {
            context.ClearTextureBindings();
            context.BindImage(0, data->mVolume);
            context.BindSampler(1, data->mSampler);
            context.BindImage(2, data->mNoise);
            context.BindImage(3, data->mLightGrid);
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
            context.SetPushConstant(data->mPcFarClip, &data->mCloudPc.mFarClip, sizeof(float));
            context.SetPushConstant(data->mPcTransmittanceLimit,
                    &data->mCloudPc.mTransmittanceLimit, sizeof(float));
            context.SetPushConstant(data->mPcTilingFreq, &data->mCloudPc.mTilingFreq,
                    sizeof(float));
            context.SetPushConstant(data->mPcDensityScale, &data->mCloudPc.mDensityScale,
                    sizeof(float));
            context.SetPushConstant(data->mPcTurbidity, &data->mCloudPc.mTurbidity,
                    sizeof(float));
            context.DrawFullscreen(*ctx.mAssets.GetProgram(data->mCloudProgram));
        });

        moe::neo::RenderTarget* hdr = data->mRenderer.GetRenderTarget(data->mHdrTarget);
        const float sunElevation = data->mCloudPc.mSunDir.y;
        const float godrayEnabled = data->mGodrayEnabled ? 1.0f : 0.0f;
        const float godrayExposure = data->mGodrayExposure;
        const float exposure = data->mExposure;
        const float whitePoint = data->mWhitePoint;
        const moe::neo::PassDesc compositePass{"nubis composite", {}, {}};
        data->mRenderer.Execute(compositePass, [&](moe::neo::PassContext& context) {
            context.ClearTextureBindings();
            context.BindImage(0, *hdr->mImage);
            context.BindSampler(1, data->mSampler);
            context.SetPushConstant(data->mPcSunScreen, &data->mSunScreen, sizeof(glm::vec2));
            context.SetPushConstant(data->mPcSunElevation, &sunElevation, sizeof(float));
            context.SetPushConstant(data->mPcGodrayEnabled, &godrayEnabled, sizeof(float));
            context.SetPushConstant(data->mPcGodrayExposure, &godrayExposure, sizeof(float));
            context.SetPushConstant(data->mPcExposure, &exposure, sizeof(float));
            context.SetPushConstant(data->mPcWhitePoint, &whitePoint, sizeof(float));
            context.DrawFullscreen(*ctx.mAssets.GetProgram(data->mCompositeProgram));
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<NubisData*>(userdata);
        ImGui::Begin("nubis demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        ImGui::SeparatorText("camera");
        ImGui::Checkbox("auto orbit", &data->mAutoOrbit);
        ImGui::SliderFloat("orbit angle", &data->mOrbitAngle, 0.0f, 6.2832f);
        ImGui::SliderFloat("orbit radius", &data->mOrbitRadius, 300.0f, 2000.0f);
        ImGui::SliderFloat("orbit height", &data->mOrbitHeight, -300.0f, 600.0f);

        ImGui::SeparatorText("cloud");
        ImGui::SliderFloat("density scale", &data->mDensityScale, 1.0f, 12.0f);
        ImGui::SliderFloat("tiling freq", &data->mTilingFreq, 0.25f, 4.0f);
        ImGui::SliderFloat("far clip", &data->mFarClip, 500.0f, 8000.0f);
        ImGui::SliderFloat("transmittance", &data->mTransmittanceLimit, 0.001f, 0.1f, "%.3f");

        ImGui::SeparatorText("environment");
        if (ImGui::SliderFloat("sun elevation", &data->mSunElevationDeg, 5.0f, 85.0f)) {
            data->mRegenerate = true;
        }
        if (ImGui::SliderFloat("sun azimuth", &data->mSunAzimuthDeg, 0.0f, 360.0f)) {
            data->mRegenerate = true;
        }
        ImGui::SliderFloat("turbidity", &data->mTurbidity, 1.0f, 20.0f);

        ImGui::SeparatorText("post");
        ImGui::Checkbox("god ray", &data->mGodrayEnabled);
        ImGui::SliderFloat("godray exposure", &data->mGodrayExposure, 0.0f, 0.2f, "%.3f");
        ImGui::SliderFloat("exposure", &data->mExposure, 0.2f, 2.0f);
        ImGui::SliderFloat("white point", &data->mWhitePoint, 0.2f, 4.0f);
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<NubisData*>(userdata);
        data->mVolumeSet.Destroy();
        data->mNoiseSet.Destroy();
        data->mLightGridSet.Destroy();
        data->mSampler.Destroy();
        data->mVolume.Destroy();
        data->mNoise.Destroy();
        data->mLightGrid.Destroy();
        data->mRenderer.DestroyRenderTarget(data->mHdrTarget);
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    NubisData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("nubis demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "nubis: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
