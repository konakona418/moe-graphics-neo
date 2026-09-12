#include "examples/common/App.hpp"

#include <Core/Error.hpp>
#include <Core/Defer.hpp>
#include <Core/Logger.hpp>
#include <Core/Profile.hpp>
#include <RHI/Image.hpp>

#include <chrono>
#include <cstdio>

namespace examples {
    // Members are default-constructed; each member's own destructor is its leak
    // trap. Run() destroys everything via scope guards, so after Run returns
    // all members are invalid and this destructor is a no-op.
    App::~App() = default;

    bool App::Run(const char* title, uint32_t width, uint32_t height,
            const AppCallbacks& callbacks) {
        MOE_PROFILE_ZONE();
        MOE_PROFILE_THREAD("main");
        moe::rhi::DeviceCreateInfo deviceInfo{};
        deviceInfo.mApplicationName = "moe-example";
        deviceInfo.mEnableValidation = true;
        deviceInfo.mEnablePresent = true;
        deviceInfo.mPipelineCache = &mPipelineCache;
        if (!moe::rhi::Device::Create(deviceInfo, mDevice)) {
            return false;
        }
        // reverse-declaration order = correct teardown order (cache, then device)
        moe::Defer deviceCleanup([&] { mDevice.Destroy(); });
        moe::Defer cacheCleanup([&] { mPipelineCache.Destroy(); });

        // clamp the requested MSAA level to what the device supports
        uint32_t sampleCount = callbacks.mSampleCount;
        if (sampleCount != 1 && sampleCount != 2 && sampleCount != 4 && sampleCount != 8) {
            moe::Logger::Warn("App: invalid sample count {}; using 4x", sampleCount);
            sampleCount = 4;
        }
        const uint32_t maxSamples = mDevice.GetMaxSampleCount();
        if (sampleCount > maxSamples) {
            moe::Logger::Warn("App: {}x MSAA unsupported; using {}x", sampleCount, maxSamples);
            sampleCount = maxSamples;
        }

        if (!mWindow.Create(mDevice, width, height, title)) {
            return false;
        }
        moe::Defer windowCleanup([&] { mWindow.Destroy(); });

        if (!mDevice.CreateSwapchain(mWindow.GetSurfaceHandle(), width, height, mSwapchain,
                    sampleCount)) {
            return false;
        }
        moe::Defer swapchainCleanup([&] { mSwapchain.Destroy(); });
        moe::Logger::Info("App: swapchain ready ({}x{})",
                mSwapchain.GetWidth(), mSwapchain.GetHeight());

        if (!mDevice.CreateCommandList(mCommandList)) {
            return false;
        }
        moe::Defer commandListCleanup([&] { mCommandList.Destroy(); });

        // Async infrastructure: the CPU pool and the GPU transfer context.
        // Registered after deviceCleanup so both are torn down before it.
        moe::Defer schedulerCleanup([&] { mScheduler.Shutdown(); });
        moe::Defer transferCleanup([&] { mTransfer.Shutdown(); });
        if (!mScheduler.Init()) {
            return false;
        }
        if (!mTransfer.Init(mDevice, mScheduler)) {
            return false;
        }

        // assets must be destroyed before the device (RHI leak traps)
        moe::Defer assetsCleanup([&] { mAssets.Destroy(); });
        if (!mAssets.Init(mDevice, mTransfer)) {
            return false;
        }

        // Input first: its GLFW callbacks are chained by whoever registers
        // later (ImGui chains the previous callbacks).
        moe::Defer inputCleanup([&] { mInput.Destroy(); });
        if (!mInput.Init(mWindow)) {
            std::fprintf(stderr, "[app] Input init failed: %s\n", moe::Error::Get().c_str());
            moe::Error::Clear();
        }

        // UI cleanup must run before the device teardown; Destroy is idempotent
        // so it is safe even when Init failed.
        moe::Defer uiCleanup([&] { mDebugUI.Destroy(); });
        if (!mDebugUI.Init(mDevice, mSwapchain, mWindow.GetHandle())) {
            std::fprintf(stderr, "[app] DebugUI init failed: %s\n", moe::Error::Get().c_str());
            moe::Error::Clear();
        } else {
            mUiActive = true;
        }

        moe::Defer im3dCleanup([&] { mIm3d.Destroy(); });
        if (!mIm3d.Init(mDevice, mPipelineCache, mSwapchain, sampleCount)) {
            std::fprintf(stderr, "[app] Im3d init failed: %s\n", moe::Error::Get().c_str());
            moe::Error::Clear();
        }

        AppContext ctx{mDevice, mPipelineCache, mWindow, mSwapchain, mIm3d, mInput, mAssets,
                mScheduler, mTransfer, sampleCount};
        if (callbacks.mSetup != nullptr && !callbacks.mSetup(callbacks.mUserdata, ctx)) {
            return moe::Fail("Setup failed");
        }

        const float defaultClear[4] = {0.15f, 0.15f, 0.18f, 1.0f};
        const float* clear = callbacks.mClearColor != nullptr ? callbacks.mClearColor : defaultClear;

        bool failed = false;
        auto lastFrame = std::chrono::steady_clock::now();
        while (!mWindow.ShouldClose()) {
            MOE_PROFILE_ZONE_NAMED("frame");
            mWindow.PollEvents();
            if (!mSwapchain.AcquireImage()) {
                continue;
            }
            const auto now = std::chrono::steady_clock::now();
            const float deltaSeconds = std::chrono::duration<float>(now - lastFrame).count();
            lastFrame = now;

            // Im3d frame: the demo fills the AppData + draws primitives.
            if (callbacks.mDrawIm3d != nullptr) {
                callbacks.mDrawIm3d(callbacks.mUserdata, ctx, deltaSeconds);
            }

            mCommandList.Begin();
            // The legacy per-frame clear pass only exists for demos that draw
            // in mRender; renderer-based demos skip it (their own passes own
            // the swapchain image layout, and Swapchain tracks it).
            if (callbacks.mRender != nullptr) {
                mSwapchain.BeginRendering(mCommandList, clear);
                callbacks.mRender(callbacks.mUserdata, ctx, mCommandList);
                mSwapchain.EndRendering(mCommandList);
            }
            if (callbacks.mPostRender != nullptr) {
                callbacks.mPostRender(callbacks.mUserdata, ctx, mCommandList);
            }

            // ImGui: new frame, record windows, composite over the swapchain
            // (in its own pass; the Swapchain tracks the image layout).
            if (mUiActive) {
                mDebugUI.BeginFrame(deltaSeconds);
                if (callbacks.mDrawUI != nullptr) {
                    callbacks.mDrawUI(callbacks.mUserdata, ctx);
                }
                if (mSwapchain.BeginRendering(mCommandList, clear, moe::rhi::LoadOp::kLoad)) {
                    mDebugUI.Render(mCommandList);
                    mSwapchain.EndRendering(mCommandList);
                }
            }

            mCommandList.End();
            if (!mSwapchain.Present(mCommandList)) {
                moe::Error::Set("Present failed");
                failed = true;
                break;
            }
            // Submit any readback copies queued this frame (after Present so
            // they execute after the frame's GPU work), then drain main-thread
            // completions posted by the transfer context's completion thread.
            mTransfer.Pump();
            mScheduler.Pump();
            mInput.EndFrame(); // clear per-frame edges + mouse deltas
            MOE_PROFILE_FRAME();
        }
        mDevice.WaitIdle();

        if (callbacks.mShutdown != nullptr) {
            callbacks.mShutdown(callbacks.mUserdata, ctx);
        }
        return !failed;
    }
}// namespace examples