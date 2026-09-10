#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>

#include <imgui.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstdlib>

namespace {
    // Minimal content-layer example: one glTF model (mesh + material texture)
    // loaded with Assets::LoadModel and drawn with DrawModel.
    struct TexturedData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::Model mModel;
        moe::neo::ProgramHandle mProgram;
        float mAngle{0.0f};
    };

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<TexturedData*>(userdata);

        data->mModel = ctx.mAssets.LoadModel(MOE_SOURCE_DIR "/examples/assets/box_textured/box.gltf");
        if (!data->mModel.IsValid()) {
            std::fprintf(stderr, "textured: import/upload: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/textured.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/textured.frag.spv");
        if (!data->mProgram.IsValid()) {
            std::fprintf(stderr, "textured: shader load: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                    ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "textured: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<TexturedData*>(userdata);
        data->mAngle += 0.01f;

        const float clear[4] = {0.1f, 0.12f, 0.15f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        moe::neo::Camera camera;
        camera.mView = glm::lookAt(glm::vec3(0.0f, 1.5f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f));
        camera.mProj = glm::perspective(glm::radians(60.0f),
                static_cast<float>(data->mFrame.GetWidth())
                        / static_cast<float>(data->mFrame.GetHeight()),
                0.1f, 100.0f);
        camera.mProj[1][1] *= -1.0f;

        const moe::neo::PassDesc pass{"model", {}, {}};
        data->mRenderer.Execute(pass, [&](moe::neo::PassContext& context) {
            context.SetCamera(camera);
            const glm::mat4 model = glm::rotate(glm::mat4(1.0f), data->mAngle,
                    glm::vec3(0.0f, 1.0f, 0.0f));
            context.DrawModel(data->mModel, data->mProgram, model);
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void*, examples::AppContext&) {
        ImGui::Begin("textured demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("glTF -> LoadModel -> DrawModel");
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<TexturedData*>(userdata);
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    TexturedData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("textured demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "textured: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
