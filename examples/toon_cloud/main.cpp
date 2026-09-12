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
        glm::vec3 mBoxMin;
        glm::vec3 mBoxMax;
        float mDensityScale;
        float mTilingFreq;
        float mFarClip;
        float mTransmittanceLimit;
        float mSteps;
        float mShadowEdge;
        float mRimStrength;
        float mHalftoneCell;
        float mHalftoneAngle;
        float mInkStrength;
    };

    // Toon-shaded clouds: the Nubis volume/light-grid compute passes author the
    // procedural cloud once, then a single Renderer fullscreen pass raymarches
    // it with cel-quantized lighting and a screen-space halftone overlay.
    struct ToonCloudData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::ProgramHandle mCloudProgram;

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
        int32_t mPcBoxMin{-1};
        int32_t mPcBoxMax{-1};
        int32_t mPcDensityScale{-1};
        int32_t mPcTilingFreq{-1};
        int32_t mPcFarClip{-1};
        int32_t mPcTransmittanceLimit{-1};
        int32_t mPcSteps{-1};
        int32_t mPcShadowEdge{-1};
        int32_t mPcRimStrength{-1};
        int32_t mPcHalftoneCell{-1};
        int32_t mPcHalftoneAngle{-1};
        int32_t mPcInkStrength{-1};

        float mDensityScale{5.0f};
        float mTilingFreq{1.0f};
        float mFarClip{4000.0f};
        float mTransmittanceLimit{0.01f};
        float mSteps{4.0f};
        float mShadowEdge{0.55f};
        float mRimStrength{1.0f};
        float mHalftoneCell{6.0f};
        float mHalftoneAngle{0.7854f};
        float mInkStrength{0.9f};
        float mSunElevationDeg{55.0f};
        float mSunAzimuthDeg{40.0f};

        bool mGenerated{false};
        bool mSunDirty{false};
        bool mAutoSun{false};
        float mSunPhase{0.0f};
        float mLastTime{0.0f};
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
            std::fprintf(stderr, "toon_cloud: volume image: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    bool SetupCompute(moe::rhi::Device& device, const char* path, moe::rhi::Shader& shader,
            moe::rhi::ShaderProgram& program, moe::rhi::ComputePipeline& pipeline,
            moe::rhi::DescriptorSetLayout& layout, moe::rhi::DescriptorSet& set) {
        if (!shader.Load(path, moe::rhi::ShaderStage::kCompute) || !program.AddShader(shader)) {
            std::fprintf(stderr, "toon_cloud: compute shader %s: %s\n", path,
                    moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::ComputePipelineState state{};
        state.mProgram = &program;
        if (!device.GetOrCreateComputePipeline(state, pipeline)
                || !pipeline.GetDescriptorSetLayout(0, layout)
                || !device.CreateDescriptorSet(layout, set)) {
            std::fprintf(stderr, "toon_cloud: compute pipeline %s: %s\n", path,
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
        auto* data = static_cast<ToonCloudData*>(userdata);

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
            std::fprintf(stderr, "toon_cloud: sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // reuse the Nubis3 volume/noise/light-grid compute shaders
        if (!SetupCompute(ctx.mDevice, MOE_SOURCE_DIR "/shaders/examples/nubis/nubis_volume.comp.spv",
                    data->mVolumeComp, data->mVolumeProgram, data->mVolumePipeline,
                    data->mVolumeSetLayout, data->mVolumeSet)
                || !SetupCompute(ctx.mDevice, MOE_SOURCE_DIR "/shaders/examples/nubis/nubis_noise.comp.spv",
                        data->mNoiseComp, data->mNoiseProgram, data->mNoisePipeline,
                        data->mNoiseSetLayout, data->mNoiseSet)
                || !SetupCompute(ctx.mDevice,
                        MOE_SOURCE_DIR "/shaders/examples/nubis/nubis_lightgrid.comp.spv",
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
            std::fprintf(stderr, "toon_cloud: descriptor writes: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mCloudProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/toon_cloud/toon_cloud.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/toon_cloud/toon_cloud.frag.spv");
        if (!data->mCloudProgram.IsValid()) {
            std::fprintf(stderr, "toon_cloud: cloud shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache, ctx.mSwapchain.GetWidth(),
                    ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "toon_cloud: renderer: %s\n", moe::Error::Get().c_str());
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
        data->mPcBoxMin = data->mRenderer.GetPushConstant(*cloud, "mBoxMin");
        data->mPcBoxMax = data->mRenderer.GetPushConstant(*cloud, "mBoxMax");
        data->mPcDensityScale = data->mRenderer.GetPushConstant(*cloud, "mDensityScale");
        data->mPcTilingFreq = data->mRenderer.GetPushConstant(*cloud, "mTilingFreq");
        data->mPcFarClip = data->mRenderer.GetPushConstant(*cloud, "mFarClip");
        data->mPcTransmittanceLimit =
                data->mRenderer.GetPushConstant(*cloud, "mTransmittanceLimit");
        data->mPcSteps = data->mRenderer.GetPushConstant(*cloud, "mSteps");
        data->mPcShadowEdge = data->mRenderer.GetPushConstant(*cloud, "mShadowEdge");
        data->mPcRimStrength = data->mRenderer.GetPushConstant(*cloud, "mRimStrength");
        data->mPcHalftoneCell = data->mRenderer.GetPushConstant(*cloud, "mHalftoneCell");
        data->mPcHalftoneAngle = data->mRenderer.GetPushConstant(*cloud, "mHalftoneAngle");
        data->mPcInkStrength = data->mRenderer.GetPushConstant(*cloud, "mInkStrength");
        if (data->mPcCameraPos < 0 || data->mPcForward < 0 || data->mPcRight < 0
                || data->mPcUp < 0 || data->mPcSunDir < 0 || data->mPcTanHalfFov < 0
                || data->mPcAspect < 0 || data->mPcBoxMin < 0 || data->mPcBoxMax < 0
                || data->mPcDensityScale < 0 || data->mPcTilingFreq < 0 || data->mPcFarClip < 0
                || data->mPcTransmittanceLimit < 0 || data->mPcSteps < 0
                || data->mPcShadowEdge < 0 || data->mPcRimStrength < 0
                || data->mPcHalftoneCell < 0 || data->mPcHalftoneAngle < 0
                || data->mPcInkStrength < 0) {
            std::fprintf(stderr, "toon_cloud: push constant names mismatch\n");
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
        data->mCloudPc.mBoxMin = kBoxMin;
        data->mCloudPc.mBoxMax = kBoxMax;
        return true;
    }

    void GenerateVolumes(ToonCloudData* data, moe::rhi::CommandList& cmd) {
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

    // The light grid is the only volume that depends on the sun, so a moving sun
    // re-runs just this dispatch (the NVDF stays in its read layout).
    void RegenerateLightGrid(ToonCloudData* data, moe::rhi::CommandList& cmd) {
        moe::rhi::SyncInfo sync{};
        sync.mSrcStage = moe::rhi::PipelineStage::kFragmentShader;
        sync.mSrcAccess = moe::rhi::Access::kShaderRead;
        sync.mDstStage = moe::rhi::PipelineStage::kComputeShader;
        sync.mDstAccess = moe::rhi::Access::kShaderWrite;
        data->mRenderer.ImageBarrier(data->mLightGrid, data->mLightGridLayout,
                moe::rhi::ImageLayout::kGeneral, sync);
        data->mLightGridLayout = moe::rhi::ImageLayout::kGeneral;

        cmd.BindDescriptorSet(data->mLightGridPipeline, data->mLightGridSet, 0);
        cmd.SetPushConstants(data->mLightGridPipeline, 0, sizeof(LightGridPushConstants),
                &data->mLightGridPc);
        cmd.Dispatch(data->mLightGridPipeline, kGridW / 8, kGridH / 8, kGridD);

        sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
        sync.mDstStage = moe::rhi::PipelineStage::kFragmentShader;
        sync.mDstAccess = moe::rhi::Access::kShaderRead;
        data->mRenderer.ImageBarrier(data->mLightGrid, moe::rhi::ImageLayout::kGeneral,
                moe::rhi::ImageLayout::kShaderReadOnly, sync);
        data->mLightGridLayout = moe::rhi::ImageLayout::kShaderReadOnly;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<ToonCloudData*>(userdata);

        static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const float time = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - start).count();
        const float delta = data->mLastTime == 0.0f ? 0.0f : time - data->mLastTime;
        data->mLastTime = time;

        // optional day cycle: sweep the sun elevation so the gold -> silver rim
        // and the dawn/dusk sky play out on their own
        if (data->mAutoSun) {
            data->mSunPhase += delta * 0.25f;
            data->mSunElevationDeg = 43.0f + 35.0f * std::sin(data->mSunPhase);
            data->mSunDirty = true;
        }

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
        data->mCloudPc.mSunDir = MakeSunDirection(data->mSunElevationDeg, data->mSunAzimuthDeg);
        data->mCloudPc.mTanHalfFov = tanHalfFov;
        data->mCloudPc.mAspect = aspect;
        data->mCloudPc.mBoxMin = kBoxMin;
        data->mCloudPc.mBoxMax = kBoxMax;
        data->mCloudPc.mDensityScale = data->mDensityScale;
        data->mCloudPc.mTilingFreq = data->mTilingFreq;
        data->mCloudPc.mFarClip = data->mFarClip;
        data->mCloudPc.mTransmittanceLimit = data->mTransmittanceLimit;
        data->mCloudPc.mSteps = data->mSteps;
        data->mCloudPc.mShadowEdge = data->mShadowEdge;
        data->mCloudPc.mRimStrength = data->mRimStrength;
        data->mCloudPc.mHalftoneCell = data->mHalftoneCell;
        data->mCloudPc.mHalftoneAngle = data->mHalftoneAngle;
        data->mCloudPc.mInkStrength = data->mInkStrength;

        const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        if (!data->mGenerated) {
            data->mLightGridPc.mSunDir = data->mCloudPc.mSunDir;
            GenerateVolumes(data, cmd);
            data->mGenerated = true;
            data->mSunDirty = false;
        } else if (data->mSunDirty) {
            data->mLightGridPc.mSunDir = data->mCloudPc.mSunDir;
            RegenerateLightGrid(data, cmd);
            data->mSunDirty = false;
        }

        const moe::neo::PassDesc cloudPass{"toon cloud", {}, {}};
        data->mRenderer.Execute(cloudPass, [&](moe::neo::PassContext& context) {
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
            context.SetPushConstant(data->mPcBoxMin, &data->mCloudPc.mBoxMin, sizeof(glm::vec3));
            context.SetPushConstant(data->mPcBoxMax, &data->mCloudPc.mBoxMax, sizeof(glm::vec3));
            context.SetPushConstant(data->mPcDensityScale, &data->mCloudPc.mDensityScale,
                    sizeof(float));
            context.SetPushConstant(data->mPcTilingFreq, &data->mCloudPc.mTilingFreq,
                    sizeof(float));
            context.SetPushConstant(data->mPcFarClip, &data->mCloudPc.mFarClip, sizeof(float));
            context.SetPushConstant(data->mPcTransmittanceLimit,
                    &data->mCloudPc.mTransmittanceLimit, sizeof(float));
            context.SetPushConstant(data->mPcSteps, &data->mCloudPc.mSteps, sizeof(float));
            context.SetPushConstant(data->mPcShadowEdge, &data->mCloudPc.mShadowEdge,
                    sizeof(float));
            context.SetPushConstant(data->mPcRimStrength, &data->mCloudPc.mRimStrength,
                    sizeof(float));
            context.SetPushConstant(data->mPcHalftoneCell, &data->mCloudPc.mHalftoneCell,
                    sizeof(float));
            context.SetPushConstant(data->mPcHalftoneAngle, &data->mCloudPc.mHalftoneAngle,
                    sizeof(float));
            context.SetPushConstant(data->mPcInkStrength, &data->mCloudPc.mInkStrength,
                    sizeof(float));
            context.DrawFullscreen(*ctx.mAssets.GetProgram(data->mCloudProgram));
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<ToonCloudData*>(userdata);
        ImGui::Begin("toon cloud demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        ImGui::SeparatorText("camera");
        ImGui::Checkbox("auto orbit", &data->mAutoOrbit);
        ImGui::SliderFloat("orbit angle", &data->mOrbitAngle, 0.0f, 6.2832f);
        ImGui::SliderFloat("orbit radius", &data->mOrbitRadius, 300.0f, 2000.0f);
        ImGui::SliderFloat("orbit height", &data->mOrbitHeight, -300.0f, 600.0f);

        ImGui::SeparatorText("toon");
        ImGui::SliderFloat("density scale", &data->mDensityScale, 1.0f, 12.0f);
        ImGui::SliderFloat("tiling freq", &data->mTilingFreq, 0.25f, 4.0f);
        ImGui::SliderFloat("far clip", &data->mFarClip, 500.0f, 8000.0f);
        ImGui::SliderFloat("transmittance", &data->mTransmittanceLimit, 0.001f, 0.1f, "%.3f");
        ImGui::SliderFloat("steps", &data->mSteps, 1.0f, 8.0f, "%.0f");
        ImGui::SliderFloat("shadow edge", &data->mShadowEdge, 0.0f, 1.0f);
        ImGui::SliderFloat("rim", &data->mRimStrength, 0.0f, 3.0f);

        ImGui::SeparatorText("halftone (screen space)");
        ImGui::SliderFloat("cell (px)", &data->mHalftoneCell, 2.0f, 24.0f);
        ImGui::SliderAngle("angle", &data->mHalftoneAngle, 0.0f, 90.0f);
        ImGui::SliderFloat("ink", &data->mInkStrength, 0.0f, 1.0f);

        ImGui::SeparatorText("environment");
        ImGui::Checkbox("auto day cycle", &data->mAutoSun);
        if (ImGui::SliderFloat("sun elevation", &data->mSunElevationDeg, 5.0f, 85.0f)) {
            data->mAutoSun = false;
            data->mSunDirty = true;
        }
        if (ImGui::SliderFloat("sun azimuth", &data->mSunAzimuthDeg, 0.0f, 360.0f)) {
            data->mSunDirty = true;
        }
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<ToonCloudData*>(userdata);
        data->mVolumeSet.Destroy();
        data->mNoiseSet.Destroy();
        data->mLightGridSet.Destroy();
        data->mSampler.Destroy();
        data->mVolume.Destroy();
        data->mNoise.Destroy();
        data->mLightGrid.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    ToonCloudData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("toon cloud demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "toon_cloud: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
