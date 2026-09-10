#include "examples/common/App.hpp"

#include <Core/Defer.hpp>
#include <RHI/Image.hpp>

#include <chrono>
#include <cstdio>

namespace examples {
    // Members are default-constructed; each member's own destructor is its leak
    // trap. Run() destroys everything via scope guards, so after Run returns
    // all members are invalid and this destructor is a no-op.
    App::~App() = default;

    bool App::Run(const char* title, uint32_t width, uint32_t height,
            const AppCallbacks& callbacks, std::string& error) {
        moe::rhi::DeviceCreateInfo deviceInfo{};
        deviceInfo.mApplicationName = "moe-example";
        deviceInfo.mEnableValidation = true;
        deviceInfo.mEnablePresent = true;
        deviceInfo.mPipelineCache = &mPipelineCache;
        if (!moe::rhi::Device::Create(deviceInfo, mDevice)) {
            error = mDevice.GetLastError();
            return false;
        }
        // reverse-declaration order = correct teardown order (cache, then device)
        moe::Defer deviceCleanup([&] { mDevice.Destroy(); });
        moe::Defer cacheCleanup([&] { mPipelineCache.Destroy(); });

        if (!mWindow.Create(mDevice, width, height, title, error)) {
            return false;
        }
        moe::Defer windowCleanup([&] { mWindow.Destroy(); });

        if (!mDevice.CreateSwapchain(mWindow.GetSurfaceHandle(), width, height, mSwapchain)) {
            error = mDevice.GetLastError();
            return false;
        }
        moe::Defer swapchainCleanup([&] { mSwapchain.Destroy(); });

        if (!mDevice.CreateCommandList(mCommandList)) {
            error = mDevice.GetLastError();
            return false;
        }
        moe::Defer commandListCleanup([&] { mCommandList.Destroy(); });

        // Input first: its GLFW callbacks are chained by whoever registers
        // later (ImGui chains the previous callbacks).
        moe::Defer inputCleanup([&] { mInput.Destroy(); });
        if (!mInput.Init(mWindow, error)) {
            std::fprintf(stderr, "[app] Input init failed: %s\n", error.c_str());
            error.clear();
        }

        // UI cleanup must run before the device teardown; Destroy is idempotent
        // so it is safe even when Init failed.
        moe::Defer uiCleanup([&] { mDebugUI.Destroy(); });
        if (!mDebugUI.Init(mDevice, mSwapchain, mWindow.GetHandle(), error)) {
            std::fprintf(stderr, "[app] DebugUI init failed: %s\n", error.c_str());
            error.clear();
        } else {
            mUiActive = true;
        }

        moe::Defer im3dCleanup([&] { mIm3d.Destroy(); });
        if (!mIm3d.Init(mDevice, mPipelineCache, mSwapchain, error)) {
            std::fprintf(stderr, "[app] Im3d init failed: %s\n", error.c_str());
            error.clear();
        }

        AppContext ctx{mDevice, mPipelineCache, mWindow, mSwapchain, mIm3d, mInput};
        if (callbacks.mSetup != nullptr && !callbacks.mSetup(callbacks.mUserdata, ctx)) {
            error = "Setup failed";
            return false;
        }

        const float defaultClear[4] = {0.15f, 0.15f, 0.18f, 1.0f};
        const float* clear = callbacks.mClearColor != nullptr ? callbacks.mClearColor : defaultClear;

        auto lastFrame = std::chrono::steady_clock::now();
        while (!mWindow.ShouldClose()) {
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
            mSwapchain.BeginRendering(mCommandList, clear);
            if (callbacks.mRender != nullptr) {
                callbacks.mRender(callbacks.mUserdata, ctx, mCommandList);
            }
            mSwapchain.EndRendering(mCommandList);
            if (callbacks.mPostRender != nullptr) {
                callbacks.mPostRender(callbacks.mUserdata, ctx, mCommandList);
            }

            // ImGui: new frame, record windows, composite over the swapchain
            // (currently in PresentSrc) in its own render pass.
            if (mUiActive) {
                mDebugUI.BeginFrame(deltaSeconds);
                if (callbacks.mDrawUI != nullptr) {
                    callbacks.mDrawUI(callbacks.mUserdata, ctx);
                }
                moe::rhi::Image swapImage;
                if (mSwapchain.GetCurrentImage(swapImage)) {
                    moe::rhi::SyncInfo uiSync{};
                    uiSync.mSrcStage = moe::rhi::PipelineStage::kBottomOfPipe;
                    uiSync.mSrcAccess = moe::rhi::Access::kNone;
                    uiSync.mDstStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
                    uiSync.mDstAccess = moe::rhi::Access::kColorAttachmentWrite;
                    mCommandList.ImageBarrier(swapImage, moe::rhi::ImageLayout::kPresentSrc,
                            moe::rhi::ImageLayout::kColorAttachment, uiSync);
                    mCommandList.BeginRendering(swapImage, clear, nullptr, 1.0f,
                            moe::rhi::LoadOp::kLoad);
                    mDebugUI.Render(mCommandList);
                    mCommandList.EndRendering();
                    uiSync.mSrcStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
                    uiSync.mSrcAccess = moe::rhi::Access::kColorAttachmentWrite;
                    uiSync.mDstStage = moe::rhi::PipelineStage::kBottomOfPipe;
                    uiSync.mDstAccess = moe::rhi::Access::kNone;
                    mCommandList.ImageBarrier(swapImage, moe::rhi::ImageLayout::kColorAttachment,
                            moe::rhi::ImageLayout::kPresentSrc, uiSync);
                    swapImage.Destroy(); // borrowed wrapper: only drops the wrapper
                }
            }

            mCommandList.End();
            if (!mSwapchain.Present(mCommandList)) {
                error = "Present failed";
                break;
            }
            mInput.EndFrame(); // clear per-frame edges + mouse deltas
        }
        mDevice.WaitIdle();

        if (callbacks.mShutdown != nullptr) {
            callbacks.mShutdown(callbacks.mUserdata, ctx);
        }
        return error.empty();
    }
}// namespace examples