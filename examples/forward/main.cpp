#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <Neo/Uploader.hpp>
#include <RHI/Shader.hpp>

#include <imgui.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
    struct ForwardData {
        moe::neo::Uploader mUploader;
        moe::neo::UploadedMesh mMesh;
        moe::rhi::Shader mVert;
        moe::rhi::Shader mFrag;
        moe::rhi::ShaderProgram mProgram;
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        int32_t mMvpIndex{-1};
        int32_t mModelIndex{-1};
        float mAngle{0.0f};
    };

    moe::neo::Mesh MakeBoxMesh() {
        const glm::vec3 corners[8] = {
                {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
                {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}};
        const glm::vec3 faceNormals[6] = {
                {0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
        const int faceCorners[6][4] = {
                {0, 1, 2, 3}, {5, 4, 7, 6}, {0, 4, 5, 1}, {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 5, 6, 2}};
        const glm::vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

        moe::neo::Mesh mesh;
        mesh.mName = "box";
        moe::neo::MeshPrimitive prim;
        for (int face = 0; face < 6; ++face) {
            for (int i = 0; i < 4; ++i) {
                const int corner = faceCorners[face][i];
                prim.mPositions.push_back(corners[corner]);
                prim.mNormals.push_back(faceNormals[face]);
                prim.mUv0.push_back(uvs[i]);
            }
            const uint32_t base = static_cast<uint32_t>(face) * 4;
            // CCW winding seen from outside the cube (Vulkan front face = CCW)
            prim.mIndices.push_back(base);
            prim.mIndices.push_back(base + 2);
            prim.mIndices.push_back(base + 1);
            prim.mIndices.push_back(base);
            prim.mIndices.push_back(base + 3);
            prim.mIndices.push_back(base + 2);
        }
        mesh.mPrimitives.push_back(std::move(prim));
        return mesh;
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<ForwardData*>(userdata);

        if (!data->mUploader.Init(ctx.mDevice)) {
            std::fprintf(stderr, "forward: uploader init: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::neo::Mesh box = MakeBoxMesh();
        if (!data->mUploader.UploadMesh(box, data->mMesh)) {
            std::fprintf(stderr, "forward: upload: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mVert.Load(MOE_SOURCE_DIR "/shaders/examples/forward/forward.vert.spv", moe::rhi::ShaderStage::kVertex)
                || !data->mFrag.Load(MOE_SOURCE_DIR "/shaders/examples/forward/forward.frag.spv", moe::rhi::ShaderStage::kFragment)) {
            std::fprintf(stderr, "forward: shader load failed\n");
            return false;
        }
        if (!data->mProgram.AddShader(data->mVert) || !data->mProgram.AddShader(data->mFrag)) {
            std::fprintf(stderr, "forward: program add failed\n");
            return false;
        }

        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "forward: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        // push constant names are chosen freely; look up once here
        data->mMvpIndex = data->mRenderer.GetPushConstant(data->mProgram, "mvp");
        data->mModelIndex = data->mRenderer.GetPushConstant(data->mProgram, "model");
        if (data->mMvpIndex < 0 || data->mModelIndex < 0) {
            std::fprintf(stderr, "forward: shader lacks 'mvp'/'model' push constants\n");
            return false;
        }
        return true;
    }

    // The scene is rendered by the Renderer in mPostRender (no render pass
    // active there); the App's swapchain pass stays empty.
    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<ForwardData*>(userdata);

        data->mAngle += 0.01f;

        const glm::mat4 model = glm::rotate(glm::mat4(1.0f), data->mAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 1.2f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                static_cast<float>(ctx.mSwapchain.GetWidth()) / static_cast<float>(ctx.mSwapchain.GetHeight()),
                0.1f, 100.0f);
        proj[1][1] *= -1; // Vulkan NDC: flip Y (same as the old engine's camera)
        const glm::mat4 mvp = proj * view * model;

        const float clear[4] = {0.15f, 0.15f, 0.18f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        const moe::neo::PassDesc mainPass{"forward", {}, {}};
        data->mRenderer.Execute(mainPass, [&](moe::neo::PassContext& pass) {
            pass.SetPushConstant(data->mMvpIndex, &mvp, sizeof(mvp));
            pass.SetPushConstant(data->mModelIndex, &model, sizeof(model));
            pass.Draw(data->mMesh, data->mProgram);
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<ForwardData*>(userdata);

        ImGui::Begin("forward demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::SliderFloat("rotation", &data->mAngle, 0.0f, 6.2832f);
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<ForwardData*>(userdata);
        data->mRenderer.Destroy();
        data->mMesh.Destroy();
    }
}// namespace

int main() {
    ForwardData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("forward demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "forward: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
