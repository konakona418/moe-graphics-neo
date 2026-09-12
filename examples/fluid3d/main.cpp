#include <examples/common/App.hpp>
#include <examples/common/Bloom.hpp>

#include <Core/Error.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Shader.hpp>

#include <imgui.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

// 3D GPU fluid (port of unity-compute-shaders 01_4_Fluid_3D): the same
// stable-fluids solver as the 2D example, on a 128^3 volume, displayed by
// raymarching the density volume (the reference's VolumeShader).

namespace {
    constexpr uint32_t kGridSize = 96;
    constexpr uint32_t kWorkgroup = 8;

    struct Kernel {
        moe::rhi::Shader mShader;
        moe::rhi::ShaderProgram mProgram;
        moe::rhi::ComputePipeline mPipeline;
        moe::rhi::DescriptorSet mSet;

        void Destroy() { mSet.Destroy(); }
    };

    bool CreateKernel(moe::rhi::Device& device, const std::string& path, Kernel& out,
            std::initializer_list<std::pair<uint32_t, moe::rhi::Image*>> images) {
        if (!out.mShader.Load(path.c_str(), moe::rhi::ShaderStage::kCompute)
                || !out.mProgram.AddShader(out.mShader)) {
            return false;
        }
        moe::rhi::ComputePipelineState state{};
        state.mProgram = &out.mProgram;
        if (!device.GetOrCreateComputePipeline(state, out.mPipeline)) {
            return false;
        }
        moe::rhi::DescriptorSetLayout layout;
        if (!out.mPipeline.GetDescriptorSetLayout(0, layout)
                || !device.CreateDescriptorSet(layout, out.mSet)) {
            return false;
        }
        for (auto& [binding, image] : images) {
            if (!out.mSet.WriteImage(binding, *image, moe::rhi::DescriptorType::kStorageImage)) {
                return false;
            }
        }
        return true;
    }

    struct ParamsPush {
        glm::vec4 mParams; // x = delta time
    };

    struct UserPush {
        glm::vec4 mSphere;   // xyz = emitter offset from centre
        glm::vec4 mVelocity; // xyz = emitter velocity
        glm::vec4 mParams;   // x = force intensity, y = force range, z = delta time
        glm::vec4 mDye;      // rgb = dye colour
    };

    struct DisplayPush {
        glm::vec4 mCameraPosTan;
        glm::vec4 mForwardAspect;
        glm::vec4 mRightOpacity;
        glm::vec4 mUpExposure;
        glm::vec4 mBoxMinSteps;
        glm::vec4 mBoxMaxPad;
    };

    struct Fluid3D {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;

        moe::rhi::Image mVelocity;
        moe::rhi::Image mDensity;
        moe::rhi::Image mPressure;
        moe::rhi::Image mDivergence;
        moe::rhi::Sampler mSampler;

        Kernel mInit;
        Kernel mDiffusion;
        Kernel mAdvection;
        Kernel mUserInput;
        Kernel mDivergenceK;
        Kernel mJacobi;
        Kernel mSubtractGradient;

        moe::rhi::Shader mDisplayVert;
        moe::rhi::Shader mDisplayFrag;
        moe::rhi::ShaderProgram mDisplayProgram;
        moe::neo::RenderTargetHandle mVolumeTarget;
        examples::Bloom mBloom;
        examples::BloomParams mBloomParams{0.5f, 0.5f, 1.5f};
        int32_t mPcCameraPosTan{-1};
        int32_t mPcForwardAspect{-1};
        int32_t mPcRightOpacity{-1};
        int32_t mPcUpExposure{-1};
        int32_t mPcBoxMinSteps{-1};
        int32_t mPcBoxMaxPad{-1};

        moe::rhi::ImageLayout mVelocityLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mDensityLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mPressureLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mDivergenceLayout{moe::rhi::ImageLayout::kUndefined};

        float mForceIntensity{200.0f};
        float mForceRange{0.10f};
        int mSolverIterations{20};
        float mOpacity{8.0f};
        float mExposure{5.0f};
        float mRaySteps{64.0f};
        bool mPaused{false};
        bool mReset{false};
        bool mInitialized{false};

        float mYaw{0.7f};
        float mPitch{0.32f};
        float mRadius{2.2f};
        bool mDragging{false};

        glm::vec3 mEmitter{0.0f};
        glm::vec3 mEmitterPrev{0.0f};
        float mMouseHold{0.0f};
        float mTime{0.0f};
        std::chrono::steady_clock::time_point mLastTime{};
    };

    glm::vec3 HsvToRgb(float h, float s, float v) {
        const float k = std::fmod(h * 6.0f, 6.0f);
        const float c = v * s;
        const float x = c * (1.0f - std::fabs(std::fmod(k, 2.0f) - 1.0f));
        glm::vec3 rgb(0.0f);
        if (k < 1.0f) {
            rgb = {c, x, 0.0f};
        } else if (k < 2.0f) {
            rgb = {x, c, 0.0f};
        } else if (k < 3.0f) {
            rgb = {0.0f, c, x};
        } else if (k < 4.0f) {
            rgb = {0.0f, x, c};
        } else if (k < 5.0f) {
            rgb = {x, 0.0f, c};
        } else {
            rgb = {c, 0.0f, x};
        }
        return rgb + glm::vec3(v - c);
    }

    bool CreateFluidImage(moe::rhi::Device& device, moe::rhi::Format format,
            moe::rhi::ImageUsage usage, moe::rhi::Image& out) {
        moe::rhi::ImageCreateInfo info{};
        info.mType = moe::rhi::ImageType::k3D;
        info.mWidth = kGridSize;
        info.mHeight = kGridSize;
        info.mDepth = kGridSize;
        info.mFormat = format;
        info.mUsage = usage;
        return device.CreateImage(info, out);
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<Fluid3D*>(userdata);

        if (!CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR32G32B32A32Float,
                    moe::rhi::ImageUsage::kStorage, data->mVelocity)
                || !CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR16G16B16A16Float,
                        moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled,
                        data->mDensity)
                || !CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR32Float,
                        moe::rhi::ImageUsage::kStorage, data->mPressure)
                || !CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR32Float,
                        moe::rhi::ImageUsage::kStorage, data->mDivergence)) {
            std::fprintf(stderr, "fluid3d: image: %s\n", moe::Error::Get().c_str());
            return false;
        }

        moe::rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mMinFilter = moe::rhi::Filter::kLinear;
        samplerInfo.mMagFilter = moe::rhi::Filter::kLinear;
        samplerInfo.mAddressModeU = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeV = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeW = moe::rhi::AddressMode::kClampToEdge;
        if (!ctx.mDevice.CreateSampler(samplerInfo, data->mSampler)) {
            std::fprintf(stderr, "fluid3d: sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const char* dir = MOE_SOURCE_DIR "/shaders/examples/fluid3d/";
        const std::string base(dir);
        if (!CreateKernel(ctx.mDevice, base + "init.comp.spv", data->mInit,
                    {{0, &data->mVelocity}, {1, &data->mDensity}, {2, &data->mPressure},
                            {3, &data->mDivergence}})
                || !CreateKernel(ctx.mDevice, base + "diffusion.comp.spv", data->mDiffusion,
                        {{0, &data->mDensity}})
                || !CreateKernel(ctx.mDevice, base + "advection.comp.spv", data->mAdvection,
                        {{0, &data->mVelocity}, {1, &data->mDensity}})
                || !CreateKernel(ctx.mDevice, base + "user_input.comp.spv", data->mUserInput,
                        {{0, &data->mVelocity}, {1, &data->mDensity}})
                || !CreateKernel(ctx.mDevice, base + "divergence.comp.spv", data->mDivergenceK,
                        {{0, &data->mVelocity}, {1, &data->mDivergence}})
                || !CreateKernel(ctx.mDevice, base + "jacobi.comp.spv", data->mJacobi,
                        {{0, &data->mPressure}, {1, &data->mDivergence}})
                || !CreateKernel(ctx.mDevice, base + "subtract_gradient.comp.spv",
                        data->mSubtractGradient, {{0, &data->mPressure}, {1, &data->mVelocity}})) {
            std::fprintf(stderr, "fluid3d: compute kernel: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mDisplayVert.Load((base + "display.vert.spv").c_str(),
                    moe::rhi::ShaderStage::kVertex)
                || !data->mDisplayFrag.Load((base + "display.frag.spv").c_str(),
                        moe::rhi::ShaderStage::kFragment)
                || !data->mDisplayProgram.AddShader(data->mDisplayVert)
                || !data->mDisplayProgram.AddShader(data->mDisplayFrag)) {
            std::fprintf(stderr, "fluid3d: display shader: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache, ctx.mSwapchain.GetWidth(),
                    ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "fluid3d: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // The volume raymarch is the frame's cost, so render it at half
        // resolution (HDR); bloom then composites it up to the swapchain.
        const uint32_t volumeWidth = std::max(1u, ctx.mSwapchain.GetWidth() / 2);
        const uint32_t volumeHeight = std::max(1u, ctx.mSwapchain.GetHeight() / 2);
        data->mVolumeTarget = data->mRenderer.CreateRenderTarget(volumeWidth, volumeHeight,
                moe::rhi::Format::kR16G16B16A16Float, false, 1);
        if (!data->mVolumeTarget.IsValid()
                || !data->mBloom.Init(ctx.mDevice, data->mRenderer,
                        MOE_SOURCE_DIR "/shaders/examples/common/", volumeWidth, volumeHeight)) {
            std::fprintf(stderr, "fluid3d: bloom: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mPcCameraPosTan = data->mRenderer.GetPushConstant(data->mDisplayProgram, "mCameraPosTan");
        data->mPcForwardAspect = data->mRenderer.GetPushConstant(data->mDisplayProgram, "mForwardAspect");
        data->mPcRightOpacity = data->mRenderer.GetPushConstant(data->mDisplayProgram, "mRightOpacity");
        data->mPcUpExposure = data->mRenderer.GetPushConstant(data->mDisplayProgram, "mUpExposure");
        data->mPcBoxMinSteps = data->mRenderer.GetPushConstant(data->mDisplayProgram, "mBoxMinSteps");
        data->mPcBoxMaxPad = data->mRenderer.GetPushConstant(data->mDisplayProgram, "mBoxMaxPad");
        if (data->mPcCameraPosTan < 0 || data->mPcForwardAspect < 0 || data->mPcRightOpacity < 0
                || data->mPcUpExposure < 0 || data->mPcBoxMinSteps < 0 || data->mPcBoxMaxPad < 0) {
            std::fprintf(stderr, "fluid3d: display push constants missing\n");
            return false;
        }

        data->mLastTime = std::chrono::steady_clock::now();
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<Fluid3D*>(userdata);

        const auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - data->mLastTime).count();
        data->mLastTime = now;
        dt = std::clamp(dt, 1.0f / 240.0f, 1.0f / 30.0f);
        if (!data->mPaused) {
            data->mTime += dt;
        }

        // ---- orbit camera: drag to rotate, scroll to zoom, slow auto-orbit ----
        const moe::neo::MouseState& mouse = ctx.mInput.GetMouse();
        const bool dragging = mouse.mButtonDown[0];
        if (dragging) {
            data->mYaw -= mouse.mDeltaX * 0.006f;
            data->mPitch = std::clamp(data->mPitch + mouse.mDeltaY * 0.006f, -1.45f, 1.45f);
        } else {
            data->mYaw += dt * 0.12f;
        }
        data->mDragging = dragging;
        data->mRadius = std::clamp(data->mRadius * (1.0f - mouse.mScrollY * 0.1f), 1.6f, 5.0f);

        const glm::vec3 target(0.5f);
        const glm::vec3 orbit(std::cos(data->mPitch) * std::sin(data->mYaw),
                std::sin(data->mPitch), std::cos(data->mPitch) * std::cos(data->mYaw));
        const glm::vec3 cameraPos = target + orbit * data->mRadius;
        const glm::vec3 forward = glm::normalize(target - cameraPos);
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 up = glm::cross(right, forward);
        const float aspect = static_cast<float>(ctx.mSwapchain.GetWidth())
                / static_cast<float>(ctx.mSwapchain.GetHeight());
        const float tanHalfFov = std::tan(glm::radians(45.0f) * 0.5f);

        // ---- emitter: auto path, overridden by recent mouse movement ----
        const float width = static_cast<float>(ctx.mSwapchain.GetWidth());
        const float height = static_cast<float>(ctx.mSwapchain.GetHeight());
        const glm::vec2 mouseUv(mouse.mX / width, mouse.mY / height);
        if (!dragging && (mouse.mDeltaX != 0.0f || mouse.mDeltaY != 0.0f)) {
            data->mMouseHold = 1.5f;
        }
        data->mMouseHold = std::max(0.0f, data->mMouseHold - dt);

        glm::vec3 target3;
        if (data->mMouseHold > 0.0f) {
            target3 = glm::vec3(mouseUv.x - 0.5f, 0.5f - mouseUv.y, 0.2f * std::sin(data->mTime * 0.5f));
        } else {
            const float t = data->mTime;
            target3 = glm::vec3(0.25f * std::sin(t * 0.50f),
                    0.20f * std::cos(t * 0.63f), 0.25f * std::sin(t * 0.41f));
        }
        target3 = glm::clamp(target3, glm::vec3(-0.42f), glm::vec3(0.42f));
        glm::vec3 velocity = target3 - data->mEmitterPrev;
        const float speed = glm::length(velocity);
        if (speed > 0.08f) {
            velocity *= 0.08f / speed;
        }
        data->mEmitter = target3;
        data->mEmitterPrev = target3;

        const float hue = 0.5f * (std::sin(data->mTime * 0.5f) + 1.0f);
        const glm::vec3 dye = HsvToRgb(hue, 1.0f, 1.0f);

        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        const float clear[4] = {0.01f, 0.012f, 0.02f, 1.0f};
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        auto toGeneral = [&](moe::rhi::Image& image, moe::rhi::ImageLayout& layout) {
            if (layout == moe::rhi::ImageLayout::kGeneral) {
                return;
            }
            moe::rhi::SyncInfo sync{};
            sync.mSrcStage = layout == moe::rhi::ImageLayout::kUndefined
                    ? moe::rhi::PipelineStage::kTopOfPipe
                    : moe::rhi::PipelineStage::kFragmentShader;
            sync.mSrcAccess = layout == moe::rhi::ImageLayout::kUndefined
                    ? moe::rhi::Access::kNone
                    : moe::rhi::Access::kShaderRead;
            sync.mDstStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mDstAccess = moe::rhi::Access::kShaderWrite;
            data->mRenderer.ImageBarrier(image, layout, moe::rhi::ImageLayout::kGeneral, sync);
            layout = moe::rhi::ImageLayout::kGeneral;
        };

        toGeneral(data->mVelocity, data->mVelocityLayout);
        toGeneral(data->mDensity, data->mDensityLayout);
        toGeneral(data->mPressure, data->mPressureLayout);
        toGeneral(data->mDivergence, data->mDivergenceLayout);

        auto dispatch = [&](Kernel& kernel, const void* pushData, size_t pushSize) {
            moe::rhi::SyncInfo sync{};
            sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
            sync.mDstStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mDstAccess = moe::rhi::Access::kShaderRead;
            data->mRenderer.MemoryBarrier(sync);
            cmd.BindDescriptorSet(kernel.mPipeline, kernel.mSet, 0);
            if (pushData != nullptr) {
                cmd.SetPushConstants(kernel.mPipeline, 0, pushSize, pushData);
            }
            cmd.Dispatch(kernel.mPipeline, kGridSize / kWorkgroup, kGridSize / kWorkgroup,
                    kGridSize / kWorkgroup);
        };

        if (data->mReset || !data->mInitialized) {
            dispatch(data->mInit, nullptr, 0);
            data->mInitialized = true;
            data->mReset = false;
        }

        if (!data->mPaused) {
            ParamsPush params{};
            params.mParams = glm::vec4(dt, 0.0f, 0.0f, 0.0f);
            dispatch(data->mDiffusion, &params, sizeof(params));
            dispatch(data->mAdvection, &params, sizeof(params));

            UserPush push{};
            push.mSphere = glm::vec4(data->mEmitter, 0.0f);
            push.mVelocity = glm::vec4(velocity, 0.0f);
            push.mParams = glm::vec4(data->mForceIntensity, data->mForceRange, dt, 0.0f);
            push.mDye = glm::vec4(dye, 0.0f);
            dispatch(data->mUserInput, &push, sizeof(push));

            dispatch(data->mDivergenceK, nullptr, 0);
            for (int i = 0; i < data->mSolverIterations; ++i) {
                dispatch(data->mJacobi, nullptr, 0);
            }
            dispatch(data->mSubtractGradient, nullptr, 0);
        }

        // Density -> ShaderReadOnly for the volume raymarch.
        {
            moe::rhi::SyncInfo sync{};
            sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
            sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
            sync.mDstStage = moe::rhi::PipelineStage::kFragmentShader;
            sync.mDstAccess = moe::rhi::Access::kShaderRead;
            data->mRenderer.ImageBarrier(data->mDensity, data->mDensityLayout,
                    moe::rhi::ImageLayout::kShaderReadOnly, sync);
            data->mDensityLayout = moe::rhi::ImageLayout::kShaderReadOnly;
        }

        DisplayPush displayPush{};
        displayPush.mCameraPosTan = glm::vec4(cameraPos, tanHalfFov);
        displayPush.mForwardAspect = glm::vec4(forward, aspect);
        displayPush.mRightOpacity = glm::vec4(right, data->mOpacity);
        displayPush.mUpExposure = glm::vec4(up, data->mExposure);
        displayPush.mBoxMinSteps = glm::vec4(0.0f, 0.0f, 0.0f, data->mRaySteps);
        displayPush.mBoxMaxPad = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);

        const moe::neo::PassDesc volumePass{"fluid3d-volume",
                moe::neo::ColorAttachment(data->mVolumeTarget, moe::rhi::LoadOp::kClear), {}};
        data->mRenderer.Execute(volumePass, [&](moe::neo::PassContext& pass) {
            pass.SetPushConstant(data->mPcCameraPosTan, &displayPush.mCameraPosTan, sizeof(glm::vec4));
            pass.SetPushConstant(data->mPcForwardAspect, &displayPush.mForwardAspect, sizeof(glm::vec4));
            pass.SetPushConstant(data->mPcRightOpacity, &displayPush.mRightOpacity, sizeof(glm::vec4));
            pass.SetPushConstant(data->mPcUpExposure, &displayPush.mUpExposure, sizeof(glm::vec4));
            pass.SetPushConstant(data->mPcBoxMinSteps, &displayPush.mBoxMinSteps, sizeof(glm::vec4));
            pass.SetPushConstant(data->mPcBoxMaxPad, &displayPush.mBoxMaxPad, sizeof(glm::vec4));
            pass.BindImage(0, data->mDensity);
            pass.BindSampler(1, data->mSampler);
            pass.DrawFullscreen(data->mDisplayProgram);
        });

        moe::neo::RenderTarget* volume = data->mRenderer.GetRenderTarget(data->mVolumeTarget);
        data->mBloom.Render(data->mRenderer, *volume->mImage, data->mBloomParams);

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<Fluid3D*>(userdata);
        ImGui::Begin("fluid 3D");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("grid: %ux%ux%u", kGridSize, kGridSize, kGridSize);
        ImGui::SliderFloat("force intensity", &data->mForceIntensity, 0.0f, 600.0f);
        ImGui::SliderFloat("force range", &data->mForceRange, 0.01f, 0.3f);
        ImGui::SliderInt("solver iterations", &data->mSolverIterations, 0, 60);
        ImGui::SliderFloat("opacity", &data->mOpacity, 0.1f, 4.0f);
        ImGui::SliderFloat("exposure", &data->mExposure, 0.1f, 8.0f);
        ImGui::SliderFloat("ray steps", &data->mRaySteps, 32.0f, 384.0f);
        ImGui::SliderFloat("bloom threshold", &data->mBloomParams.mThreshold, 0.0f, 3.0f);
        ImGui::SliderFloat("bloom intensity", &data->mBloomParams.mIntensity, 0.0f, 3.0f);
        ImGui::Checkbox("paused", &data->mPaused);
        ImGui::SameLine();
        if (ImGui::Button("reset")) {
            data->mReset = true;
        }
        ImGui::TextUnformatted("drag to orbit, scroll to zoom, move the mouse to stir");
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<Fluid3D*>(userdata);
        data->mInit.Destroy();
        data->mDiffusion.Destroy();
        data->mAdvection.Destroy();
        data->mUserInput.Destroy();
        data->mDivergenceK.Destroy();
        data->mJacobi.Destroy();
        data->mSubtractGradient.Destroy();
        data->mSampler.Destroy();
        data->mVelocity.Destroy();
        data->mDensity.Destroy();
        data->mPressure.Destroy();
        data->mDivergence.Destroy();
        data->mBloom.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    Fluid3D data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("fluid 3D", 1280, 720, callbacks)) {
        std::fprintf(stderr, "fluid3d: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
