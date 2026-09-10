#include <examples/common/App.hpp>

#include <Neo/Uploader.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Pipeline.hpp>
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
        moe::neo::UploadedMesh mMesh;
        moe::rhi::Shader mVert;
        moe::rhi::Shader mFrag;
        moe::rhi::ShaderProgram mProgram;
        moe::rhi::GraphicsPipeline mPipeline;
        float mAngle{0.0f};
    };

    struct PushConstants {
        glm::mat4 mMvp;
        glm::mat4 mModel;
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
        std::string error;

        moe::neo::Uploader uploader;
        if (!uploader.Init(ctx.mDevice, error)) {
            std::fprintf(stderr, "forward: uploader init: %s\n", error.c_str());
            return false;
        }
        moe::neo::Mesh box = MakeBoxMesh();
        if (!uploader.UploadMesh(box, data->mMesh, error)) {
            std::fprintf(stderr, "forward: upload: %s\n", error.c_str());
            return false;
        }

        if (!data->mVert.Load(MOE_SOURCE_DIR "/shaders/examples/forward.vert.spv", moe::rhi::ShaderStage::kVertex)
                || !data->mFrag.Load(MOE_SOURCE_DIR "/shaders/examples/forward.frag.spv", moe::rhi::ShaderStage::kFragment)) {
            std::fprintf(stderr, "forward: shader load failed\n");
            return false;
        }
        if (!data->mProgram.AddShader(data->mVert) || !data->mProgram.AddShader(data->mFrag)) {
            std::fprintf(stderr, "forward: program add failed\n");
            return false;
        }

        moe::rhi::GraphicsPipelineState state{};
        state.mProgram = &data->mProgram;
        state.mTopology = moe::rhi::PrimitiveTopology::kTriangleList;
        state.mColorFormatCount = 1;
        state.mColorFormats[0] = ctx.mSwapchain.GetFormat();
        state.mBlendAttachmentCount = 1;
        state.mRaster.mCullMode = moe::rhi::CullMode::kBack; // convex cube needs no depth, back-cull suffices
        state.mRaster.mFrontFace = moe::rhi::FrontFace::kCounterClockwise;
        // vertex layout matching the Uploader interleave: pos@0, nrm@12, uv@24, stride 32
        state.mVertexBindingCount = 1;
        state.mVertexBindings[0] = {0, 32, false};
        state.mVertexAttributeCount = 3;
        state.mVertexAttributes[0] = {0, 0, moe::rhi::Format::kR32G32B32Float, 0};
        state.mVertexAttributes[1] = {1, 0, moe::rhi::Format::kR32G32B32Float, 12};
        state.mVertexAttributes[2] = {2, 0, moe::rhi::Format::kR32G32Float, 24};

        if (!ctx.mDevice.GetOrCreateGraphicsPipeline(state, data->mPipeline)) {
            std::fprintf(stderr, "forward: pipeline: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }
        return true;
    }

    void Render(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<ForwardData*>(userdata);

        data->mAngle += 0.01f;

        const glm::mat4 model = glm::rotate(glm::mat4(1.0f), data->mAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 1.2f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                static_cast<float>(ctx.mSwapchain.GetWidth()) / static_cast<float>(ctx.mSwapchain.GetHeight()),
                0.1f, 100.0f);
        proj[1][1] *= -1; // Vulkan NDC: flip Y (same as the old engine's camera)
        const PushConstants pc{proj * view * model, model};

        cmd.BindGraphicsPipeline(data->mPipeline);
        cmd.SetViewport(ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight());
        cmd.BindVertexBuffer(data->mMesh.mVertexBuffer, 0);
        cmd.BindIndexBuffer(data->mMesh.mIndexBuffer);
        cmd.SetPushConstants(data->mPipeline, 0, sizeof(pc), &pc);
        cmd.DrawIndexed(data->mMesh.mIndexCount, 1, 0, 0, 0);
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
        data->mMesh.Destroy();
    }
}// namespace

int main() {
    ForwardData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mRender = Render;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    std::string error;
    if (!app.Run("forward demo", 1280, 720, callbacks, error)) {
        std::fprintf(stderr, "forward: app: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}