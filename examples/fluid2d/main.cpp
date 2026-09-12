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

// 2D GPU fluid (port of unity-compute-shaders 01_3_Fluid_2D): a stable-fluids
// solver on a 1024x1024 grid. Seven compute kernels run per step
// (diffusion -> advection -> user input -> divergence -> N x Jacobi ->
// subtract gradient); a fullscreen pass displays the dye field.

namespace {
    constexpr uint32_t kGridSize = 512;
    constexpr uint32_t kWorkgroup = 16;

    // One compute kernel: shader + program + pipeline + its descriptor set.
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

    struct FluidPush {
        glm::vec4 mSphere; // xy = emitter offset from centre, zw = velocity
        glm::vec4 mParams; // x = force intensity, y = force range, z = delta time
        glm::vec4 mDye;    // rgb = dye colour
    };

    struct Fluid2D {
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
        int32_t mPcExposure{-1};

        moe::neo::RenderTargetHandle mSceneTarget;
        examples::Bloom mBloom;
        examples::BloomParams mBloomParams{0.7f, 0.5f, 1.2f};

        moe::rhi::ImageLayout mVelocityLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mDensityLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mPressureLayout{moe::rhi::ImageLayout::kUndefined};
        moe::rhi::ImageLayout mDivergenceLayout{moe::rhi::ImageLayout::kUndefined};

        float mForceIntensity{200.0f};
        float mForceRange{0.06f};
        int mSolverIterations{30};
        float mExposure{2.5f};
        bool mPaused{false};
        bool mReset{false};
        bool mInitialized{false};

        glm::vec2 mEmitter{0.0f};
        glm::vec2 mEmitterPrev{0.0f};
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
        info.mType = moe::rhi::ImageType::k2D;
        info.mWidth = kGridSize;
        info.mHeight = kGridSize;
        info.mFormat = format;
        info.mUsage = usage;
        return device.CreateImage(info, out);
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<Fluid2D*>(userdata);

        if (!CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR32G32Float,
                    moe::rhi::ImageUsage::kStorage, data->mVelocity)
                || !CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR16G16B16A16Float,
                        moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled,
                        data->mDensity)
                || !CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR32Float,
                        moe::rhi::ImageUsage::kStorage, data->mPressure)
                || !CreateFluidImage(ctx.mDevice, moe::rhi::Format::kR32Float,
                        moe::rhi::ImageUsage::kStorage, data->mDivergence)) {
            std::fprintf(stderr, "fluid2d: image: %s\n", moe::Error::Get().c_str());
            return false;
        }

        moe::rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mMinFilter = moe::rhi::Filter::kLinear;
        samplerInfo.mMagFilter = moe::rhi::Filter::kLinear;
        samplerInfo.mAddressModeU = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeV = moe::rhi::AddressMode::kClampToEdge;
        if (!ctx.mDevice.CreateSampler(samplerInfo, data->mSampler)) {
            std::fprintf(stderr, "fluid2d: sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const char* dir = MOE_SOURCE_DIR "/shaders/examples/fluid2d/";
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
            std::fprintf(stderr, "fluid2d: compute kernel: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mDisplayVert.Load((base + "display.vert.spv").c_str(),
                    moe::rhi::ShaderStage::kVertex)
                || !data->mDisplayFrag.Load((base + "display.frag.spv").c_str(),
                        moe::rhi::ShaderStage::kFragment)
                || !data->mDisplayProgram.AddShader(data->mDisplayVert)
                || !data->mDisplayProgram.AddShader(data->mDisplayFrag)) {
            std::fprintf(stderr, "fluid2d: display shader: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache, ctx.mSwapchain.GetWidth(),
                    ctx.mSwapchain.GetHeight(), ctx.mSampleCount, ctx.mTransfer)) {
            std::fprintf(stderr, "fluid2d: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mSceneTarget = data->mRenderer.CreateRenderTarget(ctx.mSwapchain.GetWidth(),
                ctx.mSwapchain.GetHeight(), moe::rhi::Format::kR16G16B16A16Float, false);
        if (!data->mSceneTarget.IsValid()
                || !data->mBloom.Init(ctx.mDevice, data->mRenderer,
                        MOE_SOURCE_DIR "/shaders/examples/common/", ctx.mSwapchain.GetWidth(),
                        ctx.mSwapchain.GetHeight())) {
            std::fprintf(stderr, "fluid2d: bloom: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mPcExposure = data->mRenderer.GetPushConstant(data->mDisplayProgram, "exposure");
        if (data->mPcExposure < 0) {
            std::fprintf(stderr, "fluid2d: push constant 'exposure' missing\n");
            return false;
        }

        data->mLastTime = std::chrono::steady_clock::now();
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<Fluid2D*>(userdata);

        const auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - data->mLastTime).count();
        data->mLastTime = now;
        dt = std::clamp(dt, 1.0f / 240.0f, 1.0f / 30.0f);
        if (!data->mPaused) {
            data->mTime += dt;
        }

        // ---- emitter: auto path, overridden by recent mouse movement ----
        const moe::neo::MouseState& mouse = ctx.mInput.GetMouse();
        const float width = static_cast<float>(ctx.mSwapchain.GetWidth());
        const float height = static_cast<float>(ctx.mSwapchain.GetHeight());
        glm::vec2 mouseUv(mouse.mX / width, mouse.mY / height);
        if (mouse.mDeltaX != 0.0f || mouse.mDeltaY != 0.0f || mouse.mButtonDown[0]) {
            data->mMouseHold = 1.5f;
        }
        data->mMouseHold = std::max(0.0f, data->mMouseHold - dt);

        glm::vec2 target;
        if (data->mMouseHold > 0.0f) {
            target = mouseUv - 0.5f;
        } else {
            const float t = data->mTime;
            target = glm::vec2(0.30f * std::sin(t * 0.53f) + 0.08f * std::sin(t * 1.7f),
                    0.26f * std::cos(t * 0.71f) + 0.07f * std::cos(t * 1.3f));
        }
        target = glm::clamp(target, glm::vec2(-0.49f), glm::vec2(0.49f));
        glm::vec2 velocity = target - data->mEmitterPrev;
        const float speed = glm::length(velocity);
        if (speed > 0.08f) {
            velocity *= 0.08f / speed;
        }
        data->mEmitter = target;
        data->mEmitterPrev = target;

        const float hue = 0.5f * (std::sin(data->mTime * 0.5f) + 1.0f);
        const glm::vec3 dye = HsvToRgb(hue, 1.0f, 1.0f);

        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        const float clear[4] = {0.02f, 0.03f, 0.05f, 1.0f};
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        // All solver images live in General while the compute kernels run; the
        // density returns to ShaderReadOnly for the display pass each frame.
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
            cmd.Dispatch(kernel.mPipeline, kGridSize / kWorkgroup, kGridSize / kWorkgroup, 1);
        };

        if (data->mReset || !data->mInitialized) {
            dispatch(data->mInit, nullptr, 0);
            data->mInitialized = true;
            data->mReset = false;
        }

        if (!data->mPaused) {
            const glm::vec4 params(dt, 0.0f, 0.0f, 0.0f);
            dispatch(data->mDiffusion, &params, sizeof(params));
            dispatch(data->mAdvection, &params, sizeof(params));

            FluidPush push{};
            push.mSphere = glm::vec4(data->mEmitter.x, data->mEmitter.y, velocity.x, velocity.y);
            push.mParams = glm::vec4(data->mForceIntensity, data->mForceRange, dt, 0.0f);
            push.mDye = glm::vec4(dye, 0.0f);
            dispatch(data->mUserInput, &push, sizeof(push));

            dispatch(data->mDivergenceK, nullptr, 0);
            for (int i = 0; i < data->mSolverIterations; ++i) {
                dispatch(data->mJacobi, nullptr, 0);
            }
            dispatch(data->mSubtractGradient, nullptr, 0);
        }

        // Density -> ShaderReadOnly for the display pass.
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

        const moe::neo::PassDesc display{"fluid2d-scene",
                moe::neo::ColorAttachment(data->mSceneTarget, moe::rhi::LoadOp::kClear), {}};
        data->mRenderer.Execute(display, [&](moe::neo::PassContext& pass) {
            pass.SetPushConstant(data->mPcExposure, &data->mExposure, sizeof(data->mExposure));
            pass.BindImage(0, data->mDensity);
            pass.BindSampler(1, data->mSampler);
            pass.DrawFullscreen(data->mDisplayProgram);
        });

        // Bloom over the linear HDR scene, composited to the swapchain.
        moe::neo::RenderTarget* scene = data->mRenderer.GetRenderTarget(data->mSceneTarget);
        data->mBloom.Render(data->mRenderer, *scene->mImage, data->mBloomParams);

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<Fluid2D*>(userdata);
        ImGui::Begin("fluid 2D");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("grid: %ux%u", kGridSize, kGridSize);
        ImGui::SliderFloat("force intensity", &data->mForceIntensity, 0.0f, 600.0f);
        ImGui::SliderFloat("force range", &data->mForceRange, 0.005f, 0.2f);
        ImGui::SliderInt("solver iterations", &data->mSolverIterations, 0, 100);
        ImGui::SliderFloat("exposure", &data->mExposure, 0.1f, 10.0f);
        ImGui::SliderFloat("bloom threshold", &data->mBloomParams.mThreshold, 0.0f, 3.0f);
        ImGui::SliderFloat("bloom intensity", &data->mBloomParams.mIntensity, 0.0f, 3.0f);
        ImGui::Checkbox("paused", &data->mPaused);
        ImGui::SameLine();
        if (ImGui::Button("reset")) {
            data->mReset = true;
        }
        ImGui::TextUnformatted("move the mouse to stir the fluid");
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<Fluid2D*>(userdata);
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
    Fluid2D data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("fluid 2D", 1280, 720, callbacks)) {
        std::fprintf(stderr, "fluid2d: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
