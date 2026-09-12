// Declarative UI demo: a value-tree view rebuilt every frame, laid out with the
// box model and rendered into the UI's own offscreen target, then composited
// over the swapchain. Shows label / panel / button / image plus the layout
// primitives (row / column / stack / padding / spacer / align), a whole-layer
// 3D tilt (perspective view projection) and per-element Z offset.
//
// Input is decoupled: the screen pointer is unprojected onto the UI plane
// (ray/plane intersection) before it is handed to the UI, so clicks track the
// tilted layer exactly.
//
// Usage: moe-example-ui [font-file]

#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <RHI/Sampler.hpp>
#include <UI/Ui.hpp>

#include <imgui.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

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
#if defined(_WIN32)
        const char* candidates[] = {
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/segoeui.ttf",
        };
        for (const char* candidate : candidates) {
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
#else
        for (const char* pattern : {"sans-serif:lang=zh-cn", "sans-serif"}) {
            const std::string path = QueryFontconfig(pattern);
            if (!path.empty() && std::filesystem::exists(path)) {
                return path;
            }
        }
#endif
        return {};
    }

    moe::neo::Texture MakeCheckerTexture() {
        constexpr uint32_t kSize = 64;
        constexpr uint32_t kCell = 8;
        moe::neo::Texture texture;
        texture.mName = "checker";
        texture.mWidth = kSize;
        texture.mHeight = kSize;
        texture.mChannels = 4;
        texture.mSrgb = true;
        texture.mData.resize(kSize * kSize * 4);
        for (uint32_t y = 0; y < kSize; ++y) {
            for (uint32_t x = 0; x < kSize; ++x) {
                const bool light = ((x / kCell) + (y / kCell)) % 2 == 0;
                const uint8_t value = light ? 220 : 90;
                const size_t i = (y * kSize + x) * 4;
                texture.mData[i + 0] = value;
                texture.mData[i + 1] = static_cast<uint8_t>(value * 0.6f);
                texture.mData[i + 2] = static_cast<uint8_t>(value * 0.3f);
                texture.mData[i + 3] = 255;
            }
        }
        return texture;
    }

    // UI layer camera: the UI plane (z = 0, y down) sits `distance` in front of
    // a perspective camera whose field of view exactly frames the plane, so at
    // zero tilt the UI maps 1:1 to the render target. Tilt rotates the plane
    // around its center.
    struct UiCamera {
        glm::mat4 mViewProjection{1.0f};
        glm::mat4 mModel{1.0f};
    };

    UiCamera MakeUiCamera(float width, float height, float tiltX, float tiltY,
            float distance) {
        const glm::vec2 center = {width * 0.5f, height * 0.5f};
        const glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(center, 0.0f))
                * glm::rotate(glm::mat4(1.0f), glm::radians(tiltX), glm::vec3(1.0f, 0.0f, 0.0f))
                * glm::rotate(glm::mat4(1.0f), glm::radians(tiltY), glm::vec3(0.0f, 1.0f, 0.0f))
                * glm::translate(glm::mat4(1.0f), glm::vec3(-center, 0.0f));
        const glm::mat4 view = glm::lookAt(glm::vec3(center, distance),
                glm::vec3(center, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        // Fixed field of view (framed for the reference distance) so the
        // `distance` slider actually zooms the layer; deriving fovY from
        // distance would normalize it out and leave zero tilt unchanged.
        constexpr float kReferenceDistance = 1600.0f;
        const float fovY = 2.0f * std::atan(height * 0.5f / kReferenceDistance);
        const glm::mat4 proj =
                glm::perspective(fovY, width / height, 0.1f, distance * 2.0f);
        UiCamera camera;
        camera.mViewProjection = proj * view * model;
        camera.mModel = model;
        return camera;
    }

    // Screen pixels -> UI space. The UI plane (z = 0) maps to the screen through
    // a 3x3 homography; inverting it is exact and avoids near/far unprojection
    // precision.
    glm::vec2 ScreenToUi(const glm::vec2& screen, const UiCamera& camera, float width,
            float height) {
        const glm::mat4& m = camera.mViewProjection;
        glm::mat3 plane; // UI (x, y, 1) -> clip.xyz
        plane[0] = glm::vec3(m[0].x, m[0].y, m[0].w);
        plane[1] = glm::vec3(m[1].x, m[1].y, m[1].w);
        plane[2] = glm::vec3(m[3].x, m[3].y, m[3].w);
        glm::mat3 toScreen; // ndc -> screen pixels
        toScreen[0] = glm::vec3(width * 0.5f, 0.0f, 0.0f);
        toScreen[1] = glm::vec3(0.0f, height * 0.5f, 0.0f);
        toScreen[2] = glm::vec3(width * 0.5f, height * 0.5f, 1.0f);
        const glm::mat3 inverse = glm::inverse(toScreen * plane);
        const glm::vec3 ui = inverse * glm::vec3(screen.x, screen.y, 1.0f);
        return {ui.x / ui.z, ui.y / ui.z};
    }

    struct UiData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::ui::Ui mUi;
        moe::neo::Font mFont;
        moe::neo::ProgramHandle mComposite;
        moe::neo::TextureHandle mTexture;
        moe::rhi::Sampler mSampler;
        std::string mFontPath;

        int mClickCount{0};
        float mOffsetGain{60.0f}; // max layer offset in pixels
        float mTiltX{0.0f};
        float mTiltY{0.0f};
        float mDistance{1600.0f};
    };

    moe::ui::Element BuildView(UiData& data) {
        using namespace moe::ui;
        const std::string clicks = "clicks: " + std::to_string(data.mClickCount);

        Element header = Panel(
                {
                        Label("moe-ui", Style{.mFontSize = 34.0f}),
                        Label("declarative view tree + box layout"),
                },
                Style{.mBackground = glm::vec4(0.10f, 0.12f, 0.22f, 0.95f),
                        .mRadius = 12.0f, .mBorderWidth = 1.0f})
                                  .SetZ(0.0f);

        Element buttons = Row(
                {
                        Button("click me", [&data] { ++data.mClickCount; }, "demo.click")
                                .SetZ(0.35f),
                        Button("reset", [&data] { data.mClickCount = 0; }, "demo.reset")
                                .SetZ(0.35f),
                        Label(clicks).SetZ(0.35f),
                },
                Style{.mGap = 12.0f});

        Element gallery = Row(
                {
                        Image(data.mTexture).SetZ(0.8f),
                        Stack({
                                Panel({},
                                        Style{.mBackground = glm::vec4(0.16f, 0.45f, 0.75f, 1.0f),
                                                .mRadius = 8.0f})
                                        .SetZ(0.8f),
                                Label("overlaid").SetZ(0.8f),
                        }),
                },
                Style{.mGap = 12.0f});

        // Two overlapping buttons at different depths: the offset separates
        // them, and the top one's background must occlude the bottom one's text.
        Element overlap = Stack({
                Button("under", [&data] { ++data.mClickCount; }, "demo.under")
                        .SetZ(0.0f),
                Button("over", [&data] { ++data.mClickCount; }, "demo.over").SetZ(0.5f),
        });

        // Clipping + scrolling. The clipped image is larger than its box, so
        // its overflow is scissored away; the scroll view holds a tall column
        // in a fixed viewport (wheel over it).
        Element clipped = Clip(
                Image(data.mTexture).SetSize(Size::Fixed(200.0f), Size::Fixed(140.0f)),
                Style{.mBackground = glm::vec4(0.08f, 0.09f, 0.12f, 1.0f), .mRadius = 8.0f,
                        .mBorderWidth = 1.0f, .mPadding = Insets::All(8.0f)})
                                  .SetSize(Size::Fixed(120.0f), Size::Fixed(84.0f));

        Element list = Column(
                {
                        Button("scroll item 1", [&data] { ++data.mClickCount; }, "demo.scroll1"),
                        Button("scroll item 2", [&data] { ++data.mClickCount; }, "demo.scroll2"),
                        Button("scroll item 3", [&data] { ++data.mClickCount; }, "demo.scroll3"),
                        Button("scroll item 4", [&data] { ++data.mClickCount; }, "demo.scroll4"),
                        Button("scroll item 5", [&data] { ++data.mClickCount; }, "demo.scroll5"),
                        Button("scroll item 6", [&data] { ++data.mClickCount; }, "demo.scroll6"),
                },
                Style{.mGap = 6.0f});
        Element scroller = ScrollView(list,
                Style{.mBackground = glm::vec4(0.09f, 0.10f, 0.14f, 1.0f), .mRadius = 8.0f,
                        .mBorderWidth = 1.0f, .mPadding = Insets::All(10.0f)})
                                   .SetSize(Size::Fixed(260.0f), Size::Fixed(150.0f));

        const std::string gettysburg =
                "Four score and seven years ago our fathers brought forth on this continent, "
                "a new nation, conceived in Liberty, and dedicated to the proposition that "
                "all men are created equal. Now we are engaged in a great civil war, testing "
                "whether that nation, or any nation so conceived and so dedicated, can long "
                "endure. We are met on a great battle-field of that war. We have come to "
                "dedicate a portion of that field, as a final resting place for those who "
                "here gave their lives that that nation might live.";

        // kWrap: a scrollable, fixed-height viewport over the wrapped text
        // (padding >= radius keeps the text clear of the conservative scissor
        // inset; the label fills the content width so its wrap width matches).
        Element textLayouts = ScrollView(
                Label(gettysburg, Style{.mTextLayout = TextLayout::kWrap}),
                Style{.mBackground = glm::vec4(0.10f, 0.11f, 0.15f, 1.0f), .mRadius = 8.0f,
                        .mBorderWidth = 1.0f, .mPadding = Insets::All(8.0f)})
                                   .SetSize(Size::Fixed(240.0f), Size::Fixed(150.0f));

        // kEllipsis: one line, truncated with "..." (shown under the header).
        Element truncated = Align(
                Label(gettysburg, Style{.mTextLayout = TextLayout::kEllipsis})
                        .SetWidth(Size::Fixed(340.0f)),
                Alignment::kStart);

        // Clipping + scrolling row: the image is larger than its box, the
        // middle view scrolls a tall column, the right one scrolls wrapped
        // text. Align keeps the image's fixed height in the stretched row.
        Element clips = Row(
                {
                        Align(clipped, Alignment::kStart),
                        scroller,
                        textLayouts,
                },
                Style{.mGap = 16.0f});

        Element body = Column(
                {
                        header,
                        truncated,
                        buttons,
                        gallery,
                        overlap,
                        clips,
                        Align(Label("centered horizontally"), Alignment::kCenter),
                },
                Style{.mGap = 16.0f});

        // Centered so a tilt/offset excursion stays on screen. The layer
        // transform is applied by the composite, not here.
        return Padding(Align(body, Alignment::kCenter), Insets::All(24.0f));
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<UiData*>(userdata);

        data->mFont = ctx.mAssets.LoadFont(data->mFontPath.c_str(),
                "moe-ui demo declarative view tree + box layout clicks reset centered "
                "overlaid 0123456789");
        if (!data->mFont.IsValid()) {
            std::fprintf(stderr, "ui: font load: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                    ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "ui: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mUi.Init(ctx.mAssets, data->mRenderer, ctx.mDevice)) {
            std::fprintf(stderr, "ui: init: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mUi.GetTheme().mFont = data->mFont;

        data->mComposite = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/ui/ui_composite.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/ui/ui_composite.frag.spv");
        if (!data->mComposite.IsValid()) {
            std::fprintf(stderr, "ui: composite shader: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mTexture = ctx.mAssets.UploadTexture(MakeCheckerTexture());
        if (!data->mTexture.IsValid()) {
            std::fprintf(stderr, "ui: texture: %s\n", moe::Error::Get().c_str());
            return false;
        }

        moe::rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mAddressModeU = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeV = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeW = moe::rhi::AddressMode::kClampToEdge;
        if (!ctx.mDevice.CreateSampler(samplerInfo, data->mSampler)) {
            std::fprintf(stderr, "ui: sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<UiData*>(userdata);
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }

        const float clear[4] = {0.04f, 0.04f, 0.06f, 1.0f};
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        const float width = static_cast<float>(data->mFrame.GetWidth());
        const float height = static_cast<float>(data->mFrame.GetHeight());

        const UiCamera camera =
                MakeUiCamera(width, height, data->mTiltX, data->mTiltY, data->mDistance);

        moe::ui::UiFrameDesc desc;
        desc.mWidth = data->mFrame.GetWidth();
        desc.mHeight = data->mFrame.GetHeight();
        desc.mScale = 1.0f;
        // Layer transform in the UI pass (rendered at final resolution, so it
        // stays crisp); clipping is exact via the stencil.
        desc.mViewProjection = camera.mViewProjection;

        const moe::neo::MouseState& mouse = ctx.mInput.GetMouse();
        const glm::vec2 center = {width * 0.5f, height * 0.5f};
        const glm::vec2 normalized = (glm::vec2(mouse.mX, mouse.mY) - center) / center;
        desc.mOffset = normalized * data->mOffsetGain;
        desc.mInput.mPointer = ScreenToUi({mouse.mX, mouse.mY}, camera, width, height);
        desc.mInput.mPrimaryDown = mouse.mButtonDown[0];
        desc.mInput.mPrimaryPressed = mouse.mButtonPressed[0];
        desc.mInput.mPrimaryReleased = mouse.mButtonReleased[0];
        desc.mInput.mScroll = mouse.mScrollY;
        desc.mInput.mCaptured = ImGui::GetIO().WantCaptureMouse;

        data->mUi.BeginFrame(desc);
        data->mUi.EndFrame(BuildView(*data));
        data->mUi.Render(data->mRenderer);

        const moe::neo::PassDesc composite{"ui-composite", {}, {}};
        data->mRenderer.Execute(composite, [&](moe::neo::PassContext& pass) {
            moe::neo::DrawState state;
            state.mDepthTest = false;
            state.mDepthWrite = false;
            state.mCullMode = moe::rhi::CullMode::kNone;
            state.mBlendEnabled = true;
            state.mBlendSrcColor = moe::rhi::BlendFactor::kOne;
            state.mBlendDstColor = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            state.mBlendSrcAlpha = moe::rhi::BlendFactor::kOne;
            state.mBlendDstAlpha = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            pass.SetState(state);
            pass.ClearTextureBindings();
            pass.BindImage(0, data->mUi.GetImage());
            pass.BindSampler(1, data->mSampler);
            pass.DrawFullscreen(*ctx.mAssets.GetProgram(data->mComposite));
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<UiData*>(userdata);
        ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
        ImGui::Begin("moe-ui");
        ImGui::Text("%.1f fps", ImGui::GetIO().Framerate);
        ImGui::SliderFloat("tilt x (deg)", &data->mTiltX, -35.0f, 35.0f);
        ImGui::SliderFloat("tilt y (deg)", &data->mTiltY, -35.0f, 35.0f);
        ImGui::SliderFloat("distance", &data->mDistance, 700.0f, 3000.0f);
        ImGui::SliderFloat("offset gain", &data->mOffsetGain, 0.0f, 200.0f);
        ImGui::Text("clicks: %d", data->mClickCount);
        ImGui::TextWrapped("Move the pointer: the layer follows it, with nearer elements "
                           "(higher z) shifting more. The screen pointer is unprojected "
                           "onto the UI plane so clicks track the tilt. Wheel over the "
                           "scroll view to scroll its clipped contents.");
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<UiData*>(userdata);
        data->mUi.Destroy();
        data->mSampler.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main(int argc, char** argv) {
    UiData data;
    if (argc > 1) {
        data.mFontPath = argv[1];
    } else {
        data.mFontPath = FindPlatformFont();
        if (data.mFontPath.empty()) {
            std::fprintf(stderr, "ui: no system font found; pass one explicitly: "
                                 "moe-example-ui <font-file>\n");
            return EXIT_FAILURE;
        }
    }

    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("moe-ui demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "ui: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
