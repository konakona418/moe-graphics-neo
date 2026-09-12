#include <examples/common/App.hpp>

#include "Camera.hpp"
#include "Grass.hpp"
#include "Hud.hpp"
#include "Minimap.hpp"
#include "Physics.hpp"
#include "Post.hpp"
#include "Sky.hpp"
#include "Sun.hpp"
#include "Terrain.hpp"

#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>

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
    // App shell: procedural terrain + grass -> NPR post, a first-person /
    // free-fly camera with simple physics, a contour minimap and a moe::ui HUD.
    struct AppData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::RenderTargetHandle mSceneTarget;
        moe::neo::RenderTargetHandle mMinimapScene;
        moe::neo::RenderTargetHandle mMinimapTarget;

        hakoniwa::Terrain mTerrain;
        hakoniwa::TerrainParams mTerrainParams;
        hakoniwa::Sky mSky;
        hakoniwa::Grass mGrass;
        hakoniwa::GrassParams mGrassParams;
        hakoniwa::Post mPost;
        hakoniwa::PostParams mPostParams;
        hakoniwa::Minimap mMinimap;
        hakoniwa::Hud mHud;
        hakoniwa::HudState mHudState;

        hakoniwa::FreeFlyCamera mCamera;
        bool mCameraCaptured{false};

        hakoniwa::Physics mPhysics;
        bool mWalkMode{true};
        float mWalkSpeed{6.0f};
        float mEyeHeight{1.6f};

        float mSunElevationDeg{55.0f};
        float mSunAzimuthDeg{40.0f};
        float mSunPhase{0.0f};
        bool mShowDebug{true};
        float mLastTime{0.0f};
    };

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<AppData*>(userdata);

        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache, ctx.mSwapchain.GetWidth(),
                    ctx.mSwapchain.GetHeight(), ctx.mSampleCount, ctx.mTransfer)) {
            std::fprintf(stderr, "hakoniwa: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mSceneTarget = data->mRenderer.CreateRenderTarget(ctx.mSwapchain.GetWidth(),
                ctx.mSwapchain.GetHeight(), moe::rhi::Format::kR16G16B16A16Float, true, 1, false);
        data->mMinimapScene = data->mRenderer.CreateRenderTarget(
                256, 256, moe::rhi::Format::kR8G8B8A8Unorm, true, 1, false);
        data->mMinimapTarget = data->mRenderer.CreateRenderTarget(
                256, 256, moe::rhi::Format::kR8G8B8A8Unorm, false, 1, false);
        if (!data->mSceneTarget.IsValid() || !data->mMinimapScene.IsValid()
                || !data->mMinimapTarget.IsValid()) {
            std::fprintf(stderr, "hakoniwa: render targets: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mTerrain.Init(ctx.mDevice, ctx.mTransfer, ctx.mAssets, data->mRenderer,
                    data->mTerrainParams)
                || !data->mSky.Init(ctx.mAssets, data->mRenderer)
                || !data->mGrass.Init(ctx.mDevice, ctx.mTransfer, ctx.mAssets, data->mRenderer)
                || !data->mPost.Init(ctx.mDevice, ctx.mAssets, data->mRenderer)
                || !data->mMinimap.Init(ctx.mDevice, ctx.mAssets, data->mRenderer)
                || !data->mHud.Init(ctx.mDevice, ctx.mAssets, data->mRenderer)) {
            return false;
        }

        hakoniwa::PhysicsParams physicsParams;
        physicsParams.mTerrainSize = data->mTerrainParams.mSize;
        physicsParams.mTerrainAmplitude = data->mTerrainParams.mAmplitude;
        if (!data->mPhysics.Init(physicsParams, glm::vec3(0.0f, 15.0f, 20.0f))) {
            return false;
        }
        data->mCamera.mPosition =
                data->mPhysics.PlayerPosition() + glm::vec3(0.0f, data->mEyeHeight, 0.0f);
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<AppData*>(userdata);

        static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const float time = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - start).count();
        const float delta = data->mLastTime == 0.0f ? 0.0f : time - data->mLastTime;
        data->mLastTime = time;
        const float dt = glm::clamp(delta, 0.0f, 0.05f);

        // Auto day cycle (HUD toggle).
        if (data->mHudState.mAutoCycle) {
            data->mSunPhase += delta * 0.25f;
            data->mSunElevationDeg = 43.0f + 35.0f * std::sin(data->mSunPhase);
        }

        // right-drag captures the mouse (shared by walk and fly)
        const ImGuiIO& io = ImGui::GetIO();
        const moe::neo::MouseState& mouse = ctx.mInput.GetMouse();
        if (data->mCameraCaptured) {
            if (!mouse.mButtonDown[1]
                    || ctx.mInput.IsKeyJustPressed(
                            static_cast<int32_t>(moe::neo::KeyCode::kEscape))) {
                ctx.mInput.SetMouseCaptured(false);
                data->mCameraCaptured = false;
            }
        } else if (!io.WantCaptureMouse && mouse.mButtonDown[1]) {
            ctx.mInput.SetMouseCaptured(true);
            data->mCameraCaptured = true;
        }
        if (ctx.mInput.IsKeyJustPressed(static_cast<int32_t>(moe::neo::KeyCode::kV))) {
            data->mWalkMode = !data->mWalkMode;
        }

        if (data->mWalkMode) {
            data->mCamera.Look(ctx.mInput, data->mCameraCaptured);
            const float yaw = data->mCamera.mYaw;
            const glm::vec3 forward(std::sin(yaw), 0.0f, -std::cos(yaw));
            const glm::vec3 right(std::cos(yaw), 0.0f, std::sin(yaw));
            glm::vec3 direction(0.0f);
            if (ctx.mInput.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kW))) {
                direction += forward;
            }
            if (ctx.mInput.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kS))) {
                direction -= forward;
            }
            if (ctx.mInput.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kD))) {
                direction += right;
            }
            if (ctx.mInput.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kA))) {
                direction -= right;
            }
            if (glm::length(direction) > 0.0f) {
                direction = glm::normalize(direction);
            }
            const bool jump =
                    ctx.mInput.IsKeyJustPressed(static_cast<int32_t>(moe::neo::KeyCode::kSpace));
            data->mPhysics.Update(dt, direction * data->mWalkSpeed, jump);
            data->mCamera.mPosition = data->mPhysics.PlayerPosition()
                    + glm::vec3(0.0f, data->mEyeHeight, 0.0f);
        } else {
            data->mCamera.Update(ctx.mInput, dt, data->mCameraCaptured);
        }

        // photo mode / debug toggles (290 = GLFW_KEY_F1)
        if (!io.WantCaptureKeyboard) {
            if (ctx.mInput.IsKeyJustPressed(static_cast<int32_t>(moe::neo::KeyCode::kP))) {
                data->mHudState.mPhotoMode = !data->mHudState.mPhotoMode;
                data->mHudState.mShowMenu = false;
            }
            if (ctx.mInput.IsKeyJustPressed(290)) {
                data->mShowDebug = !data->mShowDebug;
            }
        }
        if (data->mHudState.mResetCamera) {
            data->mCamera = hakoniwa::FreeFlyCamera{};
            data->mHudState.mResetCamera = false;
        }
        data->mHudState.mWalkMode = data->mWalkMode;
        data->mHudState.mSunElevationDeg = data->mSunElevationDeg;
        data->mHudState.mSunAzimuthDeg = data->mSunAzimuthDeg;

        const float aspect = static_cast<float>(ctx.mSwapchain.GetWidth())
                / static_cast<float>(ctx.mSwapchain.GetHeight());
        const glm::vec3 sunDir = hakoniwa::MakeSunDirection(
                data->mSunElevationDeg, data->mSunAzimuthDeg);
        const glm::vec3 sunColor = hakoniwa::SunColorFor(sunDir);
        const glm::vec3 fogColor = hakoniwa::SkyHorizonFor(sunDir);
        const float dusk = 1.0f - glm::smoothstep(0.02f, 0.5f, sunDir.y);
        const glm::vec3 zenithColor = glm::mix(
                glm::vec3(0.16f, 0.34f, 0.66f), glm::vec3(0.10f, 0.13f, 0.28f), dusk);
        const glm::vec3 ambient = glm::vec3(0.35f, 0.38f, 0.44f) + 0.35f * sunColor;
        const glm::mat4 viewProj = data->mCamera.Projection(aspect) * data->mCamera.View();

        const float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        hakoniwa::SkyFrame skyFrame;
        skyFrame.mForward = data->mCamera.Forward();
        skyFrame.mRight = data->mCamera.Right();
        skyFrame.mUp = data->mCamera.Up();
        skyFrame.mTanHalfFov = data->mCamera.TanHalfFov();
        skyFrame.mAspect = aspect;
        skyFrame.mSunDir = sunDir;
        skyFrame.mSunColor = sunColor;
        skyFrame.mHorizonColor = fogColor;
        skyFrame.mZenithColor = zenithColor;
        data->mSky.Record(ctx.mAssets, data->mRenderer, data->mSceneTarget, skyFrame);

        hakoniwa::TerrainFrame terrainFrame;
        terrainFrame.mViewProj = viewProj;
        terrainFrame.mCameraPos = data->mCamera.mPosition;
        terrainFrame.mSunDir = sunDir;
        terrainFrame.mSunColor = sunColor;
        terrainFrame.mAmbient = ambient;
        terrainFrame.mFogColor = fogColor;
        data->mTerrain.Record(ctx.mAssets, data->mRenderer, data->mSceneTarget, terrainFrame,
                data->mTerrainParams);

        hakoniwa::GrassFrame grassFrame;
        grassFrame.mViewProj = viewProj;
        grassFrame.mCameraPos = data->mCamera.mPosition;
        grassFrame.mSunDir = sunDir;
        grassFrame.mSunColor = sunColor;
        grassFrame.mAmbient = ambient;
        grassFrame.mFogColor = fogColor;
        grassFrame.mFogDensity = data->mTerrainParams.mFogDensity;
        grassFrame.mTerrainAmplitude = data->mTerrainParams.mAmplitude;
        grassFrame.mTerrainHalfSize = data->mTerrainParams.mSize * 0.5f;
        grassFrame.mPlayerPos = data->mPhysics.PlayerPosition();
        grassFrame.mDeltaTime = dt;
        grassFrame.mTime = time;
        data->mGrass.Record(cmd, ctx.mAssets, data->mRenderer, data->mSceneTarget, grassFrame,
                data->mGrassParams);

        data->mPost.Record(ctx.mAssets, data->mRenderer, data->mSceneTarget, data->mSceneTarget,
                moe::neo::RenderTargetHandle{}, ctx.mSwapchain.GetWidth(),
                ctx.mSwapchain.GetHeight(), data->mCamera.mNear, data->mCamera.mFar,
                data->mPostParams);

        // minimap: top-down terrain depth -> contour image, shown in the HUD.
        data->mTerrain.RecordMinimap(
                ctx.mAssets, data->mRenderer, data->mMinimapScene, data->mTerrainParams);
        data->mMinimap.Record(ctx.mAssets, data->mRenderer, data->mMinimapScene,
                data->mMinimapTarget, data->mTerrainParams.mAmplitude);

        data->mHud.Record(ctx.mAssets, data->mRenderer, ctx.mInput, data->mHudState,
                data->mMinimapTarget, ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight());

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<AppData*>(userdata);
        if (!data->mShowDebug) {
            return;
        }
        ImGui::Begin("hakoniwa");
        ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);

        ImGui::SeparatorText("camera");
        ImGui::Text("pos %.1f %.1f %.1f", data->mCamera.mPosition.x, data->mCamera.mPosition.y,
                data->mCamera.mPosition.z);
        ImGui::Checkbox("walk mode (V)", &data->mWalkMode);
        ImGui::SliderFloat("walk speed", &data->mWalkSpeed, 1.0f, 20.0f);
        ImGui::SliderFloat("speed", &data->mCamera.mSpeed, 5.0f, 400.0f);
        ImGui::SliderFloat("fov", &data->mCamera.mFovYDeg, 30.0f, 90.0f);

        ImGui::SeparatorText("sun / sky");
        ImGui::Checkbox("auto day cycle", &data->mHudState.mAutoCycle);
        ImGui::SliderFloat("sun elevation", &data->mSunElevationDeg, 5.0f, 85.0f);
        ImGui::SliderFloat("sun azimuth", &data->mSunAzimuthDeg, 0.0f, 360.0f);

        ImGui::SeparatorText("terrain");
        ImGui::SliderFloat("fog density", &data->mTerrainParams.mFogDensity, 0.0f, 0.004f, "%.5f");

        ImGui::SeparatorText("grass");
        ImGui::SliderFloat("base tile size", &data->mGrassParams.mBaseTileSize, 0.5f, 8.0f);
        ImGui::SliderInt("tiles per side", &data->mGrassParams.mTilesPerSide, 8, 64);
        ImGui::SliderInt("blades per tile", &data->mGrassParams.mBladesPerTile, 4, 192);
        ImGui::SliderInt("rings", &data->mGrassParams.mRingCount, 1, 6);
        ImGui::SliderFloat("wind", &data->mGrassParams.mWindStrength, 0.0f, 3.0f);

        ImGui::SeparatorText("NPR post");
        ImGui::SliderFloat("toon levels", &data->mPostParams.mToonLevels, 2.0f, 10.0f, "%.0f");
        ImGui::SliderFloat("halftone cell", &data->mPostParams.mHalftoneCell, 2.0f, 24.0f);
        ImGui::SliderAngle("halftone angle", &data->mPostParams.mHalftoneAngle, 0.0f, 90.0f);
        ImGui::SliderFloat("ink", &data->mPostParams.mInk, 0.0f, 1.0f);
        ImGui::SliderFloat("outline", &data->mPostParams.mOutlineStrength, 0.0f, 1.0f);
        ImGui::SliderFloat("outline threshold", &data->mPostParams.mOutlineThreshold, 0.005f, 0.2f,
                "%.3f");
        ImGui::SliderFloat(
                "grass outline", &data->mPostParams.mGrassOutlineSuppress, 0.0f, 1.0f);

        ImGui::Separator();
        ImGui::Checkbox("main menu", &data->mHudState.mShowMenu);
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<AppData*>(userdata);
        data->mHud.Destroy();
        data->mMinimap.Destroy();
        data->mPost.Destroy();
        data->mGrass.Destroy();
        data->mSky.Destroy();
        data->mTerrain.Destroy();
        data->mPhysics.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    AppData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;
    callbacks.mSampleCount = 1;

    examples::App app;
    if (!app.Run("hakoniwa", 1600, 900, callbacks)) {
        std::fprintf(stderr, "hakoniwa: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
