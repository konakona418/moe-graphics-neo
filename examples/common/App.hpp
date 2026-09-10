#pragma once

#include <Neo/Assets.hpp>
#include <Neo/Input.hpp>
#include <Neo/Window.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Device.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Swapchain.hpp>
#include <UI/DebugUI.hpp>
#include <UI/Im3dDrawer.hpp>

#include <chrono>
#include <string>

namespace examples {
    struct AppContext {
        moe::rhi::Device& mDevice;
        moe::rhi::DefaultPipelineCache& mPipelineCache;
        moe::neo::Window& mWindow;
        moe::rhi::Swapchain& mSwapchain;
        moe::ui::Im3dDrawer& mIm3d;
        moe::neo::Input& mInput;
        moe::neo::Assets& mAssets;
        // Effective MSAA level (clamped to device support); pass it to
        // neo::Renderer::Init so the renderer and swapchain agree.
        uint32_t mSampleCount{1};
    };

    struct AppCallbacks {
        // Called once after device/window/swapchain are ready. Return false to abort.
        bool (*mSetup)(void* userdata, AppContext& ctx) = nullptr;
        // Called every frame, between the swapchain's BeginRendering/EndRendering.
        void (*mRender)(void* userdata, AppContext& ctx, moe::rhi::CommandList& cmd) = nullptr;
        // Called every frame after the swapchain's EndRendering but before
        // Present. No render pass is active: use it to record offscreen passes
        // (RenderGraph) and final copies/blits into the swapchain image.
        void (*mPostRender)(void* userdata, AppContext& ctx, moe::rhi::CommandList& cmd) = nullptr;
        // Called every frame after ImGui::NewFrame; record ImGui windows here.
        // The App composites the UI over the swapchain automatically.
        void (*mDrawUI)(void* userdata, AppContext& ctx) = nullptr;
        // Called every frame at the start (before recording): the demo sets
        // Im3d::GetAppData() (camera + input), calls Im3d::NewFrame(), draws
        // primitives and Im3d::EndFrame(). The demo's render pass then calls
        // ctx.mIm3d.UploadVertices()/Record().
        void (*mDrawIm3d)(void* userdata, AppContext& ctx, float deltaSeconds) = nullptr;
        // Called after the loop ends, while resources are still alive.
        void (*mShutdown)(void* userdata, AppContext& ctx) = nullptr;
        void* mUserdata = nullptr;
        const float* mClearColor = nullptr; // RGBA; nullptr = dark gray
        // Requested MSAA level (1/2/4/8). The App clamps it to the device's
        // supported count and exposes the result as AppContext::mSampleCount.
        uint32_t mSampleCount{4};
    };

    // Windowed app: owns device + cache + window + swapchain + frame loop.
    // Lifecycle is handled internally via scope guards; after Run returns the
    // app needs no explicit teardown.
    class App {
    public:
        App() = default;
        ~App();

        App(const App&) = delete;
        App& operator=(const App&) = delete;

        bool Run(const char* title, uint32_t width, uint32_t height,
                const AppCallbacks& callbacks);

    private:
        moe::rhi::Device mDevice;
        moe::rhi::DefaultPipelineCache mPipelineCache;
        moe::neo::Window mWindow;
        moe::rhi::Swapchain mSwapchain;
        moe::rhi::CommandList mCommandList;
        moe::neo::Input mInput;
        moe::neo::Assets mAssets;
        moe::ui::DebugUI mDebugUI;
        moe::ui::Im3dDrawer mIm3d;
        bool mUiActive{false};
        std::chrono::steady_clock::time_point mFpsTime{};
        uint32_t mFpsFrames{0};
    };
}// namespace examples