#include <examples/common/App.hpp>

#include <Neo/Importer.hpp>
#include <Neo/Uploader.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
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
    struct TexturedData {
        moe::neo::UploadedMesh mMesh;
        moe::neo::UploadedTexture mTexture;
        moe::rhi::Shader mVert;
        moe::rhi::Shader mFrag;
        moe::rhi::ShaderProgram mProgram;
        moe::rhi::GraphicsPipeline mPipeline;
        moe::rhi::DescriptorSetLayout mSetLayout;
        moe::rhi::DescriptorSet mSet;
        float mAngle{0.0f};
    };

    struct PushConstants {
        glm::mat4 mMvp;
        glm::mat4 mModel;
    };

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<TexturedData*>(userdata);
        std::string error;

        // full chain: glTF (with an external PNG) -> importer decode -> upload
        moe::neo::Scene scene;
        if (!moe::neo::ImportGltf(MOE_SOURCE_DIR "/examples/assets/box_textured/box.gltf",
                scene, error)) {
            std::fprintf(stderr, "textured: import: %s\n", error.c_str());
            return false;
        }
        if (scene.mMeshes.empty() || scene.mTextures.empty()
                || scene.mTextures[0].mData.empty()) {
            std::fprintf(stderr, "textured: import produced no mesh/texture\n");
            return false;
        }
        std::printf("textured: imported '%s' (%ux%u %s)\n", scene.mName.c_str(),
                static_cast<unsigned>(scene.mTextures[0].mWidth),
                static_cast<unsigned>(scene.mTextures[0].mHeight),
                scene.mTextures[0].mSrgb ? "srgb" : "linear");

        moe::neo::Uploader uploader;
        if (!uploader.Init(ctx.mDevice, error)) {
            std::fprintf(stderr, "textured: uploader init: %s\n", error.c_str());
            return false;
        }
        if (!uploader.UploadMesh(scene.mMeshes[0], data->mMesh, error)) {
            std::fprintf(stderr, "textured: mesh upload: %s\n", error.c_str());
            return false;
        }
        if (!uploader.UploadTexture(scene.mTextures[0], data->mTexture, error)) {
            std::fprintf(stderr, "textured: texture upload: %s\n", error.c_str());
            return false;
        }

        if (!data->mVert.Load(MOE_SOURCE_DIR "/shaders/examples/textured.vert.spv", moe::rhi::ShaderStage::kVertex)
                || !data->mFrag.Load(MOE_SOURCE_DIR "/shaders/examples/textured.frag.spv", moe::rhi::ShaderStage::kFragment)) {
            std::fprintf(stderr, "textured: shader load failed\n");
            return false;
        }
        if (!data->mProgram.AddShader(data->mVert) || !data->mProgram.AddShader(data->mFrag)) {
            std::fprintf(stderr, "textured: program add failed\n");
            return false;
        }

        moe::rhi::GraphicsPipelineState state{};
        state.mProgram = &data->mProgram;
        state.mTopology = moe::rhi::PrimitiveTopology::kTriangleList;
        state.mColorFormatCount = 1;
        state.mColorFormats[0] = ctx.mSwapchain.GetFormat();
        state.mBlendAttachmentCount = 1;
        state.mRaster.mCullMode = moe::rhi::CullMode::kBack;
        state.mRaster.mFrontFace = moe::rhi::FrontFace::kCounterClockwise;
        // vertex layout matching the Uploader interleave: pos@0, nrm@12, uv@24, stride 32
        state.mVertexBindingCount = 1;
        state.mVertexBindings[0] = {0, 32, false};
        state.mVertexAttributeCount = 3;
        state.mVertexAttributes[0] = {0, 0, moe::rhi::Format::kR32G32B32Float, 0};
        state.mVertexAttributes[1] = {1, 0, moe::rhi::Format::kR32G32B32Float, 12};
        state.mVertexAttributes[2] = {2, 0, moe::rhi::Format::kR32G32Float, 24};

        if (!ctx.mDevice.GetOrCreateGraphicsPipeline(state, data->mPipeline)) {
            std::fprintf(stderr, "textured: pipeline: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }

        // sampled image (binding 0) + separate sampler (binding 1)
        if (!data->mPipeline.GetDescriptorSetLayout(0, data->mSetLayout)) {
            std::fprintf(stderr, "textured: no descriptor set layout 0\n");
            return false;
        }
        if (!ctx.mDevice.CreateDescriptorSet(data->mSetLayout, data->mSet)) {
            std::fprintf(stderr, "textured: descriptor set: %s\n", ctx.mDevice.GetLastError().c_str());
            return false;
        }
        if (!data->mSet.WriteImage(0, data->mTexture.mImage, moe::rhi::DescriptorType::kSampledImage)
                || !data->mSet.WriteSampler(1, data->mTexture.mSampler)) {
            std::fprintf(stderr, "textured: descriptor write failed\n");
            return false;
        }
        return true;
    }

    void Render(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<TexturedData*>(userdata);

        data->mAngle += 0.01f;

        const glm::mat4 model = glm::rotate(glm::mat4(1.0f), data->mAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 1.2f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                static_cast<float>(ctx.mSwapchain.GetWidth()) / static_cast<float>(ctx.mSwapchain.GetHeight()),
                0.1f, 100.0f);
        proj[1][1] *= -1; // Vulkan NDC: flip Y
        const PushConstants pc{proj * view * model, model};

        cmd.BindGraphicsPipeline(data->mPipeline);
        cmd.SetViewport(ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight());
        cmd.BindDescriptorSet(data->mPipeline, data->mSet, 0);
        cmd.SetPushConstants(data->mPipeline, 0, sizeof(pc), &pc);
        cmd.BindVertexBuffer(data->mMesh.mVertexBuffer, 0);
        cmd.BindIndexBuffer(data->mMesh.mIndexBuffer);
        cmd.DrawIndexed(data->mMesh.mIndexCount, 1, 0, 0, 0);
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<TexturedData*>(userdata);

        ImGui::Begin("textured demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::SliderFloat("rotation", &data->mAngle, 0.0f, 6.2832f);
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<TexturedData*>(userdata);
        data->mSet.Destroy();
        data->mTexture.Destroy();
        data->mMesh.Destroy();
    }
}// namespace

int main() {
    TexturedData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mRender = Render;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    std::string error;
    if (!app.Run("textured demo", 1280, 720, callbacks, error)) {
        std::fprintf(stderr, "textured: app: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
