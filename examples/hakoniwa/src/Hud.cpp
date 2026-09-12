#include "Hud.hpp"

#include "Sun.hpp"

#include <Core/Error.hpp>

#include <imgui.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// HUD module: a moe::ui declarative overlay on a tilted, pointer-parallax
// layer, composited over the post result.

namespace hakoniwa {
    namespace {
        std::string QueryFontconfig(const char* pattern) {
            const std::string command =
                    std::string("fc-match -f '%{file}' '") + pattern + "' 2>/dev/null";
            FILE* pipe = popen(command.c_str(), "r");
            if (pipe == nullptr) {
                return {};
            }
            char buffer[1024] = {};
            const char* line = std::fgets(buffer, sizeof(buffer), pipe);
            const int status = pclose(pipe);
            if (line == nullptr || status != 0) {
                return {};
            }
            std::string path(buffer);
            while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) {
                path.pop_back();
            }
            return path;
        }

        std::string FindPlatformFont() {
            for (const char* pattern : {"sans-serif:lang=zh-cn", "sans-serif"}) {
                const std::string path = QueryFontconfig(pattern);
                if (!path.empty() && std::filesystem::exists(path)) {
                    return path;
                }
            }
            return {};
        }

        // The UI plane (z = 0, y down) gets a fixed 3D deflection; the pointer
        // parallax is the depth-scaled layer offset (UiFrameDesc::mOffset).
        struct HudCamera {
            glm::mat4 mViewProjection{1.0f};
        };

        HudCamera MakeHudCamera(float width, float height, float tiltX, float tiltY) {
            const glm::vec2 center = {width * 0.5f, height * 0.5f};
            const glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(center, 0.0f))
                    * glm::rotate(glm::mat4(1.0f), glm::radians(tiltX),
                            glm::vec3(1.0f, 0.0f, 0.0f))
                    * glm::rotate(glm::mat4(1.0f), glm::radians(tiltY),
                            glm::vec3(0.0f, 1.0f, 0.0f))
                    * glm::translate(glm::mat4(1.0f), glm::vec3(-center, 0.0f));
            constexpr float kDistance = 1600.0f;
            const glm::mat4 view = glm::lookAt(glm::vec3(center, kDistance),
                    glm::vec3(center, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            const float fovY = 2.0f * std::atan(height * 0.5f / kDistance);
            const glm::mat4 proj =
                    glm::perspective(fovY, width / height, 0.1f, kDistance * 2.0f);
            HudCamera camera;
            camera.mViewProjection = proj * view * model;
            return camera;
        }

        glm::vec2 ScreenToUi(const glm::vec2& screen, const glm::mat4& viewProj, float width,
                float height) {
            glm::mat3 plane;
            plane[0] = glm::vec3(viewProj[0].x, viewProj[0].y, viewProj[0].w);
            plane[1] = glm::vec3(viewProj[1].x, viewProj[1].y, viewProj[1].w);
            plane[2] = glm::vec3(viewProj[3].x, viewProj[3].y, viewProj[3].w);
            glm::mat3 toScreen;
            toScreen[0] = glm::vec3(width * 0.5f, 0.0f, 0.0f);
            toScreen[1] = glm::vec3(0.0f, height * 0.5f, 0.0f);
            toScreen[2] = glm::vec3(width * 0.5f, height * 0.5f, 1.0f);
            const glm::mat3 inverse = glm::inverse(toScreen * plane);
            const glm::vec3 ui = inverse * glm::vec3(screen.x, screen.y, 1.0f);
            return {ui.x / ui.z, ui.y / ui.z};
        }

        // The declarative HUD tree, rebuilt every frame. Pure ASCII text.
        moe::ui::Element BuildHud(HudState& state, moe::neo::Renderer& renderer,
                moe::neo::RenderTargetHandle minimapTarget, const moe::rhi::Sampler& minimapSampler,
                float width, float height) {
            using namespace moe::ui;
            const glm::vec3 sunDir =
                    MakeSunDirection(state.mSunElevationDeg, state.mSunAzimuthDeg);

            const float dayT = glm::clamp((sunDir.y + 0.1f) / 0.9f, 0.0f, 1.0f);
            const float hours = 6.0f + 12.0f * dayT;
            char clock[16];
            std::snprintf(clock, sizeof(clock), "%02d:%02d", static_cast<int>(hours),
                    static_cast<int>((hours - static_cast<float>(static_cast<int>(hours)))
                            * 60.0f));

            const Size panelWidth = Size::Fixed(400.0f);
            const float z = 0.45f;
            const Style cardStyle{.mBackground = glm::vec4(0.10f, 0.10f, 0.11f, 0.85f),
                    .mRadius = 12.0f, .mBorderWidth = 1.0f, .mPadding = Insets::All(26.0f),
                    .mGap = 10.0f};
            const Style keyStyle{.mTextColor = glm::vec4(0.93f, 0.93f, 0.94f, 0.95f),
                    .mFontSize = 24.0f};

            std::vector<Element> items;
            if (state.mShowControls) {
                items.push_back(Label("controls",
                        Style{.mTextColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.96f),
                                .mFontSize = 56.0f})
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Spacer(Size::Fixed(8.0f)));
                items.push_back(Label("WASD   move", keyStyle).SetWidth(panelWidth).SetZ(z));
                items.push_back(Label("space   jump", keyStyle).SetWidth(panelWidth).SetZ(z));
                items.push_back(Label("right-drag   look", keyStyle).SetWidth(panelWidth).SetZ(z));
                items.push_back(Label("V   walk / fly", keyStyle).SetWidth(panelWidth).SetZ(z));
                items.push_back(Label("P   photo mode", keyStyle).SetWidth(panelWidth).SetZ(z));
                items.push_back(Label("F1   debug", keyStyle).SetWidth(panelWidth).SetZ(z));
                items.push_back(Spacer(Size::Fixed(12.0f)));
                items.push_back(Button("Back",
                        [&state] {
                            state.mShowControls = false;
                            state.mShowMenu = true;
                        },
                        "menu.back")
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
            } else if (state.mShowMenu) {
                items.push_back(Label("hakoniwa",
                        Style{.mTextColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.96f),
                                .mFontSize = 68.0f})
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Spacer(Size::Fixed(12.0f)));
                items.push_back(Button("Enter", [&state] { state.mShowMenu = false; }, "menu.enter")
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Button("Photo mode",
                        [&state] {
                            state.mShowMenu = false;
                            state.mPhotoMode = true;
                        },
                        "menu.photo")
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Button("Controls",
                        [&state] { state.mShowControls = true; }, "menu.controls")
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Button("Auto day cycle",
                        [&state] { state.mAutoCycle = !state.mAutoCycle; }, "menu.daycycle")
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Button("Reset camera",
                        [&state] { state.mResetCamera = true; }, "menu.reset")
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
            } else if (state.mPhotoMode) {
                items.push_back(Label("photo mode",
                        Style{.mTextColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.9f),
                                .mFontSize = 52.0f})
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Label("press P to exit", keyStyle).SetWidth(panelWidth).SetZ(z));
            } else {
                items.push_back(Panel({
                        Label("hakoniwa", Style{.mFontSize = 46.0f}).SetZ(z),
                        Label(state.mWalkMode ? "walk" : "free-fly", Style{.mFontSize = 22.0f})
                                .SetZ(z),
                },
                        cardStyle)
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                items.push_back(Panel({
                        Label(clock, Style{.mFontSize = 44.0f}).SetZ(z),
                        Label(state.mAutoCycle ? "auto day cycle" : "manual sun",
                                Style{.mFontSize = 20.0f})
                                .SetZ(z),
                },
                        cardStyle)
                                      .SetWidth(panelWidth)
                                      .SetZ(z));
                if (moe::neo::RenderTarget* target = renderer.GetRenderTarget(minimapTarget)) {
                    items.push_back(Panel({
                            Image(*target->mImage, minimapSampler)
                                    .SetSize(Size::Fixed(320.0f), Size::Fixed(320.0f))
                                    .SetZ(z),
                    },
                            cardStyle)
                                          .SetWidth(panelWidth)
                                          .SetZ(z));
                }
                items.push_back(Label("P: photo   F1: debug", keyStyle)
                                        .SetWidth(panelWidth)
                                        .SetZ(z));
            }

            ColumnData menuData;
            menuData.mChildren = std::move(items);
            menuData.mGap = 10.0f;
            menuData.mAlign = Alignment::kStart;
            Element menu(menuData);
            menu.SetWidth(panelWidth);
            return Column({
                    Spacer(),
                    Row({Spacer(), menu, Spacer(Size::Fixed(140.0f))}),
                    Spacer(),
            }).SetSize(Size::Fixed(width), Size::Fixed(height));
        }
    }// namespace

    bool Hud::Init(moe::rhi::Device& device, moe::neo::Assets& assets,
            moe::neo::Renderer& renderer) {
        const std::string fontPath = FindPlatformFont();
        if (!fontPath.empty()) {
            mFont = assets.LoadFont(fontPath.c_str(), "hakoniwa0123456789:.%()-+ ");
        }
        if (!mUi.Init(assets, renderer, device)) {
            std::fprintf(stderr, "hakoniwa: ui: %s\n", moe::Error::Get().c_str());
            return false;
        }
        mUi.GetTheme().mFont = mFont;
        mUi.GetTheme().mFontSize = 26.0f;
        mUi.GetTheme().mTextColor = glm::vec4(0.94f, 0.94f, 0.95f, 1.0f);
        mUi.GetTheme().mButtonColor = glm::vec4(0.93f, 0.93f, 0.94f, 0.94f);
        mUi.GetTheme().mButtonHover = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        mUi.GetTheme().mButtonActive = glm::vec4(0.76f, 0.76f, 0.78f, 1.0f);
        mUi.GetTheme().mButtonText = glm::vec4(0.09f, 0.09f, 0.10f, 1.0f);
        mUi.GetTheme().mBorderColor = glm::vec4(0.50f, 0.50f, 0.52f, 1.0f);
        mUi.GetTheme().mRadius = 8.0f;
        mUi.GetTheme().mPadding = moe::ui::Insets::Symmetric(20.0f, 10.0f);

        mComposite = assets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/ui/ui_composite.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/ui/ui_composite.frag.spv");
        if (!mComposite.IsValid()) {
            std::fprintf(stderr, "hakoniwa: hud composite: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mAddressModeU = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeV = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeW = moe::rhi::AddressMode::kClampToEdge;
        if (!device.CreateSampler(samplerInfo, mSampler)) {
            std::fprintf(stderr, "hakoniwa: hud sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    void Hud::Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
            const moe::neo::Input& input, HudState& state,
            moe::neo::RenderTargetHandle minimapTarget, uint32_t width, uint32_t height) {
        if (!mFont.IsValid()) {
            return;
        }
        const float hudWidth = static_cast<float>(width);
        const float hudHeight = static_cast<float>(height);
        const moe::neo::MouseState& mouse = input.GetMouse();
        moe::ui::UiFrameDesc hud{};
        hud.mWidth = width;
        hud.mHeight = height;
        hud.mScale = 1.0f;
        const glm::vec2 center(hudWidth * 0.5f, hudHeight * 0.5f);
        const glm::vec2 normalized = (glm::vec2(mouse.mX, mouse.mY) - center) / center;
        constexpr float kTiltX = 0.0f;
        constexpr float kTiltY = -20.0f;
        const HudCamera camera = MakeHudCamera(hudWidth, hudHeight, kTiltX, kTiltY);
        hud.mViewProjection = camera.mViewProjection;
        hud.mOffset = normalized * 25.0f;
        hud.mInput.mPointer =
                ScreenToUi({mouse.mX, mouse.mY}, camera.mViewProjection, hudWidth, hudHeight);
        hud.mInput.mPrimaryDown = mouse.mButtonDown[0];
        hud.mInput.mPrimaryPressed = mouse.mButtonPressed[0];
        hud.mInput.mPrimaryReleased = mouse.mButtonReleased[0];
        hud.mInput.mScroll = mouse.mScrollY;
        hud.mInput.mCaptured = ImGui::GetIO().WantCaptureMouse;

        mUi.BeginFrame(hud);
        mUi.EndFrame(BuildHud(state, renderer, minimapTarget, mSampler, hudWidth, hudHeight));
        mUi.Render(renderer);

        const moe::neo::PassDesc pass{"hakoniwa hud",
                moe::neo::ColorAttachment({}, moe::rhi::LoadOp::kLoad), {}};
        renderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState drawState;
            drawState.mDepthTest = false;
            drawState.mDepthWrite = false;
            drawState.mCullMode = moe::rhi::CullMode::kNone;
            drawState.mBlendEnabled = true;
            drawState.mBlendSrcColor = moe::rhi::BlendFactor::kOne;
            drawState.mBlendDstColor = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            drawState.mBlendSrcAlpha = moe::rhi::BlendFactor::kOne;
            drawState.mBlendDstAlpha = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            context.SetState(drawState);
            context.ClearTextureBindings();
            context.BindImage(0, mUi.GetImage());
            context.BindSampler(1, mSampler);
            context.DrawFullscreen(*assets.GetProgram(mComposite));
        });
    }

    void Hud::Destroy() {
        mUi.Destroy();
        mSampler.Destroy();
    }
}// namespace hakoniwa
