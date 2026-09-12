// Text rendering demo.
//
// Glyph outlines are rendered directly on the GPU (no font atlas): text
// stays crisp at any zoom and rotates/shears freely. Mouse wheel zooms
// around the cursor, middle-drag pans, 0 resets the view.
//
// Usage: moe-example-text [font-file]
// Without an argument the platform font is used (a CJK-capable sans-serif
// when one is installed); the demo exits when no font can be resolved.

#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Font.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>

#include <imgui.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {
    const char kSampleZh[] =
            "燕子去了，有再来的时候；\n"
            "杨柳枯了，有再青的时候；\n"
            "桃花谢了，有再开的时候。\n"
            "但是，聪明的，你告诉我，\n"
            "我们的日子为什么一去不复返呢？\n";

    const char kSampleJa[] =
            "吾輩は猫である。\n"
            "名前はまだ無い。\n"
            "どこで生れたかとんと見当がつかぬ。\n"
            "何でも薄暗いじめじめした所で\n"
            "ニャーニャー泣いていた事だけは記憶している。\n";

    const char kSampleEn[] =
            "Four score and seven years ago\n"
            "our fathers brought forth on this continent,\n"
            "a new nation, conceived in Liberty,\n"
            "and dedicated to the proposition that all men are created equal.\n";

    // Asks fontconfig for a font file; empty when fc-match is unavailable or
    // returns nothing usable.
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

    // Resolves the demo font without hardcoded paths: a known Windows font
    // on Windows, fontconfig elsewhere.
    std::string FindPlatformFont() {
#if defined(_WIN32)
        const char* candidates[] = {
                "C:/Windows/Fonts/msyh.ttc",   // Microsoft YaHei
                "C:/Windows/Fonts/simsun.ttc", // SimSun
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

    struct TextData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::Font mFont;
        moe::neo::ProgramHandle mProgram;
        std::string mFontPath;
        bool mHasCjk{false};

        float mPixelSize{28.0f};
        float mLetterSpacing{0.0f};
        float mLineSpacing{0.0f};
        glm::vec4 mDemoColor{1.0f, 0.65f, 0.25f, 1.0f};

        glm::vec2 mPan{0.0f};
        float mZoom{1.0f};
        float mRotation{0.0f};
        float mRotationSpeed{60.0f};
        bool mPanning{false};

        uint32_t mGlyphCount{0};
        uint32_t mCurveCount{0};
    };

    void ResetView(TextData& data) {
        data.mPan = {0.0f, 0.0f};
        data.mZoom = 1.0f;
    }

    void UpdateInput(TextData& data, examples::AppContext& ctx) {
        const moe::neo::MouseState& mouse = ctx.mInput.GetMouse();
        if (!ImGui::GetIO().WantCaptureMouse) {
            if (mouse.mScrollY != 0.0f) {
                const float oldZoom = data.mZoom;
                data.mZoom = glm::clamp(data.mZoom * std::pow(1.15f, mouse.mScrollY),
                        0.05f, 40.0f);
                // Keep the text point under the cursor fixed while zooming.
                data.mPan += glm::vec2(mouse.mX, mouse.mY)
                        * (1.0f / oldZoom - 1.0f / data.mZoom);
            }
            if (mouse.mButtonPressed[2]) {
                data.mPanning = true;
            }
            if (data.mPanning && mouse.mButtonDown[2]) {
                data.mPan -= glm::vec2(mouse.mDeltaX, mouse.mDeltaY) / data.mZoom;
            }
            if (mouse.mButtonReleased[2]) {
                data.mPanning = false;
            }
        }
        if (ctx.mInput.IsKeyJustPressed(static_cast<int32_t>(moe::neo::KeyCode::k0))) {
            ResetView(data);
        }
    }

    void DrawParagraph(moe::neo::PassContext& context, TextData& data, const char* text,
            float x, float y, const glm::vec4& color, float pixelSize) {
        moe::neo::TextDrawParams params;
        params.mPixelSize = pixelSize;
        params.mColor = color;
        params.mLetterSpacing = data.mLetterSpacing;
        params.mLineSpacing = data.mLineSpacing;
        params.mTransform = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, 0.0f));
        context.DrawText(data.mFont, text, params, data.mProgram);
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<TextData*>(userdata);

        const std::string sample = std::string(kSampleZh) + kSampleJa + kSampleEn;
        data->mFont = ctx.mAssets.LoadFont(data->mFontPath.c_str(), sample);
        if (!data->mFont.IsValid()) {
            std::fprintf(stderr, "text: font load: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::neo::FontData* font = data->mFont.GetData();
        data->mGlyphCount = static_cast<uint32_t>(font->mGlyphs.size());
        data->mCurveCount = static_cast<uint32_t>(font->mCurves.size() / 8);
        data->mHasCjk = font->FindGlyph(U'的') != nullptr;
        if (!data->mHasCjk) {
            std::printf("text: '%s' has no CJK glyphs; pass a CJK font path for the "
                        "Chinese/Japanese samples\n",
                    data->mFontPath.c_str());
        }

        data->mProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/text/text.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/text/text.frag.spv");
        if (!data->mProgram.IsValid()) {
            std::fprintf(stderr, "text: shader load: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                    ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount, ctx.mTransfer)) {
            std::fprintf(stderr, "text: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<TextData*>(userdata);
        UpdateInput(*data, ctx);
        data->mRotation += data->mRotationSpeed * ImGui::GetIO().DeltaTime;
        if (data->mRotation >= 360.0f) {
            data->mRotation -= 360.0f;
        }

        const float clear[4] = {0.05f, 0.05f, 0.07f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        const float width = static_cast<float>(data->mFrame.GetWidth());
        const float height = static_cast<float>(data->mFrame.GetHeight());

        // Pixel-space orthographic camera: text space is y down, matching
        // the font layout, and maps directly to Vulkan's y-down NDC.
        moe::neo::Camera camera;
        camera.mProj = glm::ortho(0.0f, width, 0.0f, height, -1.0f, 1.0f);
        camera.mView = glm::scale(glm::mat4(1.0f), glm::vec3(data->mZoom, data->mZoom, 1.0f))
                * glm::translate(glm::mat4(1.0f),
                        glm::vec3(-data->mPan.x, -data->mPan.y, 0.0f));

        const moe::neo::PassDesc pass{"text", {}, {}};
        data->mRenderer.Execute(pass, [&](moe::neo::PassContext& context) {
            context.SetCamera(camera);

            moe::neo::DrawState state;
            state.mDepthTest = false;
            state.mDepthWrite = false;
            state.mCullMode = moe::rhi::CullMode::kNone;
            // The text shader outputs premultiplied coverage.
            state.mBlendEnabled = true;
            state.mBlendSrcColor = moe::rhi::BlendFactor::kOne;
            state.mBlendDstColor = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            state.mBlendSrcAlpha = moe::rhi::BlendFactor::kOne;
            state.mBlendDstAlpha = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            context.SetState(state);

            if (data->mHasCjk) {
                DrawParagraph(context, *data, kSampleZh, 25.0f, 40.0f,
                        glm::vec4(1.0f, 1.0f, 0.2f, 1.0f), data->mPixelSize);
                DrawParagraph(context, *data, kSampleJa, 25.0f, 240.0f,
                        glm::vec4(0.2f, 1.0f, 1.0f, 1.0f), data->mPixelSize);
                DrawParagraph(context, *data, kSampleEn, 25.0f, 440.0f,
                        glm::vec4(1.0f, 0.71f, 0.76f, 1.0f), data->mPixelSize);
            } else {
                DrawParagraph(context, *data, kSampleEn, 25.0f, 40.0f,
                        glm::vec4(1.0f, 0.71f, 0.76f, 1.0f), data->mPixelSize);
            }

            // Rotating demo glyph, pivoting around its own center.
            moe::neo::FontData* font = data->mFont.GetData();
            const uint32_t demoCodepoint = data->mHasCjk ? U'的' : U'O';
            const moe::neo::Glyph* glyph =
                    font != nullptr ? font->FindGlyph(demoCodepoint) : nullptr;
            if (glyph != nullptr) {
                const float scale = font->GetScaleForPixelHeight(120.0f);
                // Text space puts the first baseline at ascent * scale; the
                // glyph's geometric center is measured from there.
                const float baseline = font->mAscent * scale;
                const glm::vec2 center = {
                        (glyph->mBoundsMin.x + glyph->mBoundsMax.x) * 0.5f * scale,
                        baseline - (glyph->mBoundsMin.y + glyph->mBoundsMax.y) * 0.5f * scale,
                };
                moe::neo::TextDrawParams params;
                params.mPixelSize = 120.0f;
                params.mColor = data->mDemoColor;
                params.mTransform = glm::translate(glm::mat4(1.0f),
                                           glm::vec3(width - 150.0f, 180.0f, 0.0f))
                        * glm::rotate(glm::mat4(1.0f), glm::radians(data->mRotation),
                                glm::vec3(0.0f, 0.0f, 1.0f))
                        * glm::translate(glm::mat4(1.0f), glm::vec3(-center.x, -center.y, 0.0f));
                context.DrawText(data->mFont,
                        data->mHasCjk ? "\u7684" : "O", params, data->mProgram);
            }
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<TextData*>(userdata);
        ImGui::Begin("text rendering");
        ImGui::Text("%.1f fps", ImGui::GetIO().Framerate);
        ImGui::SliderFloat("pixel size", &data->mPixelSize, 8.0f, 160.0f);
        ImGui::SliderFloat("letter spacing", &data->mLetterSpacing, -10.0f, 40.0f);
        ImGui::SliderFloat("line spacing", &data->mLineSpacing, -20.0f, 60.0f);
        ImGui::SliderFloat("rotation speed", &data->mRotationSpeed, 0.0f, 360.0f);
        ImGui::ColorEdit4("demo color", &data->mDemoColor.x);
        ImGui::Text("zoom: %.2f", data->mZoom);
        ImGui::Text("font: %s", data->mFontPath.c_str());
        ImGui::Text("glyphs: %u, curves: %u%s", data->mGlyphCount, data->mCurveCount,
                data->mHasCjk ? "" : " (no CJK)");
        if (ImGui::Button("reset view")) {
            ResetView(*data);
        }
        ImGui::Text("wheel: zoom | middle drag: pan | 0: reset");
        ImGui::Text("usage: moe-example-text [font-file]");
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<TextData*>(userdata);
        data->mRenderer.Destroy();
    }
}// namespace

int main(int argc, char** argv) {
    TextData data;
    if (argc > 1) {
        data.mFontPath = argv[1];
    } else {
        data.mFontPath = FindPlatformFont();
        if (data.mFontPath.empty()) {
            std::fprintf(stderr, "text: no system font found; pass one explicitly: "
                                 "moe-example-text <font-file>\n");
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
    if (!app.Run("text rendering demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "text: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
