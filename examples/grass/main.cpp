#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <Neo/TextureLoader.hpp>
#include <Neo/Uploader.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Shader.hpp>

#include <imgui.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
    constexpr uint32_t kGrassCount = 4000;
    constexpr uint32_t kNoiseSize = 512;

    struct PostfxData;

    // ---- CRTP passes: structured, type-bound pass classes ----

    struct ScenePass : moe::neo::Pass<ScenePass> {
        static constexpr const char* kName = "scene";
        moe::neo::RenderTargetHandle mTarget;
        moe::rhi::LoadOp mLoadOp{moe::rhi::LoadOp::kClear};
        PostfxData* mData{nullptr};

        void Execute(moe::neo::PassContext& pass);
    };

    struct BlitPass : moe::neo::Pass<BlitPass> {
        static constexpr const char* kName = "composite";
        moe::neo::RenderTargetHandle mTarget; // invalid = swapchain
        moe::rhi::LoadOp mLoadOp{moe::rhi::LoadOp::kClear};
        PostfxData* mData{nullptr};

        void Execute(moe::neo::PassContext& pass);
    };

    struct PostfxData {
        moe::neo::Uploader mUploader;
        moe::neo::UploadedMesh mBoxMesh;
        moe::neo::UploadedMesh mGrassMesh;
        moe::neo::UploadedMesh mGroundMesh;
        moe::neo::UploadedTexture mBoxTexture;
        moe::rhi::Buffer mInstanceBuffer;
        moe::rhi::Shader mSceneVert;
        moe::rhi::Shader mSceneFrag;
        moe::rhi::ShaderProgram mSceneProgram;
        moe::rhi::Shader mGrassVert;
        moe::rhi::Shader mGrassFrag;
        moe::rhi::ShaderProgram mGrassProgram;
        moe::rhi::Shader mGroundVert;
        moe::rhi::Shader mGroundFrag;
        moe::rhi::ShaderProgram mGroundProgram;
        moe::rhi::Shader mBlitVert;
        moe::rhi::Shader mBlitFrag;
        moe::rhi::ShaderProgram mBlitProgram;
        moe::rhi::Shader mNoiseComp;
        moe::rhi::ShaderProgram mNoiseProgram;
        moe::rhi::ComputePipeline mNoisePipeline;
        moe::rhi::DescriptorSet mNoiseSet;
        moe::rhi::Image mNoiseImage;
        moe::rhi::Sampler mNoiseSampler;
        moe::rhi::ImageLayout mNoiseLayout{moe::rhi::ImageLayout::kUndefined};
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::RenderTargetHandle mSceneTarget;

        int32_t mSceneMvp{-1};
        int32_t mSceneModel{-1};
        int32_t mGrassMvp{-1};
        int32_t mGrassTime{-1};
        int32_t mGroundMvp{-1};
        int32_t mBlitDisplace{-1};
        int32_t mBlitChromatic{-1};
        int32_t mBlitVignette{-1};
        int32_t mBlitScanline{-1};

        float mDisplace{0.02f};
        float mChromatic{0.004f};
        float mVignette{0.25f};
        float mScanline{0.15f};
        float mTime{0.0f};

        ScenePass mScenePass;
        BlitPass mBlitPass;
    };

    void ScenePass::Execute(moe::neo::PassContext& pass) {
        PostfxData& data = *mData;

        const float angle = data.mTime * 0.6f;
        const glm::mat4 boxModel = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 5.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                static_cast<float>(data.mFrame.GetWidth()) / static_cast<float>(data.mFrame.GetHeight()),
                0.1f, 100.0f);
        proj[1][1] *= -1;
        const glm::mat4 vp = proj * view;

        // ground (grid quad)
        pass.SetPushConstant(data.mGroundMvp, &vp, sizeof(vp));
        pass.Draw(data.mGroundMesh, data.mGroundProgram);

        // textured box
        const glm::mat4 boxMvp = vp * boxModel;
        pass.SetPushConstant(data.mSceneMvp, &boxMvp, sizeof(boxMvp));
        pass.SetPushConstant(data.mSceneModel, &boxModel, sizeof(boxModel));
        pass.BindImage(0, data.mBoxTexture.mImage);
        pass.BindSampler(1, data.mBoxTexture.mSampler);
        pass.Draw(data.mBoxMesh, data.mSceneProgram);

        // instanced grass: per-instance float4x4 (slang assigns locations
        // 1..4 to the mat4, right after position@0); bent by the noise
        pass.SetPushConstant(data.mGrassMvp, &vp, sizeof(vp));
        pass.SetPushConstant(data.mGrassTime, &data.mTime, sizeof(data.mTime));
        pass.BindImage(0, data.mNoiseImage);
        pass.BindSampler(1, data.mNoiseSampler);
        const moe::neo::InstanceAttribute attrs[4] = {
                {1, moe::rhi::Format::kR32G32B32A32Float, 0},
                {2, moe::rhi::Format::kR32G32B32A32Float, 16},
                {3, moe::rhi::Format::kR32G32B32A32Float, 32},
                {4, moe::rhi::Format::kR32G32B32A32Float, 48},
        };
        pass.BindInstanceBuffer(data.mInstanceBuffer, sizeof(glm::mat4), attrs, 4);
        pass.Draw(data.mGrassMesh, data.mGrassProgram,
                moe::rhi::PrimitiveTopology::kTriangleList, kGrassCount);
    }

    void BlitPass::Execute(moe::neo::PassContext& pass) {
        PostfxData& data = *mData;
        moe::neo::RenderTarget* scene = data.mRenderer.GetRenderTarget(data.mSceneTarget);
        pass.SetPushConstant(data.mBlitDisplace, &data.mDisplace, sizeof(data.mDisplace));
        pass.SetPushConstant(data.mBlitChromatic, &data.mChromatic, sizeof(data.mChromatic));
        pass.SetPushConstant(data.mBlitVignette, &data.mVignette, sizeof(data.mVignette));
        pass.SetPushConstant(data.mBlitScanline, &data.mScanline, sizeof(data.mScanline));
        pass.BindImage(0, *scene->mImage);
        pass.BindSampler(1, data.mNoiseSampler);
        pass.BindImage(2, data.mNoiseImage);
        pass.BindSampler(3, data.mNoiseSampler);
        pass.DrawFullscreen(data.mBlitProgram);
    }

    // ---- data setup ----

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
        mesh.mName = "grass_box";
        moe::neo::MeshPrimitive prim;
        for (int face = 0; face < 6; ++face) {
            for (int i = 0; i < 4; ++i) {
                const int corner = faceCorners[face][i];
                prim.mPositions.push_back(corners[corner]);
                prim.mNormals.push_back(faceNormals[face]);
                prim.mUv0.push_back(uvs[i]);
            }
            const uint32_t base = static_cast<uint32_t>(face) * 4;
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

    moe::neo::Mesh MakeGrassMesh() {
        moe::neo::Mesh mesh;
        mesh.mName = "grass_blade";
        moe::neo::MeshPrimitive prim;
        // a single blade: wide base at the ground, tip up top
        prim.mPositions.push_back(glm::vec3(-0.03f, 0.0f, 0.0f));
        prim.mPositions.push_back(glm::vec3(0.03f, 0.0f, 0.0f));
        prim.mPositions.push_back(glm::vec3(0.0f, 0.35f, 0.0f));
        prim.mIndices.push_back(0);
        prim.mIndices.push_back(1);
        prim.mIndices.push_back(2);
        mesh.mPrimitives.push_back(std::move(prim));
        return mesh;
    }

    moe::neo::Mesh MakeGroundMesh() {
        moe::neo::Mesh mesh;
        mesh.mName = "grass_ground";
        moe::neo::MeshPrimitive prim;
        // a single large quad (20x20), repeating uv 0..20 for grid lines
        const float half = 10.0f;
        const glm::vec3 corners[4] = {
                {-half, 0.0f, -half}, {half, 0.0f, -half}, {half, 0.0f, half}, {-half, 0.0f, half}};
        const glm::vec2 uvs[4] = {{0, 0}, {20, 0}, {20, 20}, {0, 20}};
        for (int i = 0; i < 4; ++i) {
            prim.mPositions.push_back(corners[i]);
            prim.mUv0.push_back(uvs[i]);
        }
        prim.mIndices.push_back(0);
        prim.mIndices.push_back(2);
        prim.mIndices.push_back(1);
        prim.mIndices.push_back(0);
        prim.mIndices.push_back(3);
        prim.mIndices.push_back(2);
        mesh.mPrimitives.push_back(std::move(prim));
        return mesh;
    }

    moe::neo::Texture MakeCheckerTexture() {
        moe::neo::Texture texture;
        texture.mName = "grass_checker";
        texture.mWidth = 16;
        texture.mHeight = 16;
        texture.mDepth = 1;
        texture.mChannels = 4;
        texture.mSrgb = true;
        texture.mData.resize(16 * 16 * 4);
        for (uint32_t y = 0; y < 16; ++y) {
            for (uint32_t x = 0; x < 16; ++x) {
                const bool on = ((x / 4) + (y / 4)) % 2 == 0;
                const uint8_t c = on ? 200 : 60;
                const size_t i = (y * 16 + x) * 4;
                texture.mData[i + 0] = c;
                texture.mData[i + 1] = c;
                texture.mData[i + 2] = c;
                texture.mData[i + 3] = 255;
            }
        }
        return texture;
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<PostfxData*>(userdata);

        if (!data->mUploader.Init(ctx.mDevice)) {
            std::fprintf(stderr, "grass: uploader: %s\n", moe::Error::Get().c_str());
            return false;
        }

        if (!data->mUploader.UploadMesh(MakeBoxMesh(), data->mBoxMesh)
                || !data->mUploader.UploadMesh(MakeGrassMesh(), data->mGrassMesh)
                || !data->mUploader.UploadMesh(MakeGroundMesh(), data->mGroundMesh)) {
            std::fprintf(stderr, "grass: mesh upload: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mUploader.UploadTexture(MakeCheckerTexture(), data->mBoxTexture)) {
            std::fprintf(stderr, "grass: texture upload: %s\n", moe::Error::Get().c_str());
            return false;
        }

        std::vector<glm::mat4> instances(kGrassCount);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> posDist(-5.0f, 5.0f);
        std::uniform_real_distribution<float> rotDist(0.0f, 6.2832f);
        std::uniform_real_distribution<float> scaleDist(0.6f, 1.2f);
        for (auto& m : instances) {
            m = glm::mat4(1.0f);
            m = glm::translate(m, glm::vec3(posDist(rng), 0.0f, posDist(rng)));
            m = glm::rotate(m, rotDist(rng), glm::vec3(0.0f, 1.0f, 0.0f));
            const float s = scaleDist(rng);
            m = glm::scale(m, glm::vec3(s, s, s));
        }
        if (!data->mUploader.UploadData(reinterpret_cast<const uint8_t*>(instances.data()),
                instances.size() * sizeof(glm::mat4), moe::rhi::BufferUsage::kVertex,
                data->mInstanceBuffer)) {
            std::fprintf(stderr, "grass: instance upload: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const char* shaderDir = MOE_SOURCE_DIR "/shaders/examples/";
        if (!data->mSceneVert.Load((std::string(shaderDir) + "grass_scene.vert.spv").c_str(), moe::rhi::ShaderStage::kVertex)
                || !data->mSceneFrag.Load((std::string(shaderDir) + "grass_scene.frag.spv").c_str(), moe::rhi::ShaderStage::kFragment)
                || !data->mSceneProgram.AddShader(data->mSceneVert)
                || !data->mSceneProgram.AddShader(data->mSceneFrag)) {
            std::fprintf(stderr, "grass: scene shader load failed\n");
            return false;
        }
        if (!data->mGrassVert.Load((std::string(shaderDir) + "grass_blade.vert.spv").c_str(), moe::rhi::ShaderStage::kVertex)
                || !data->mGrassFrag.Load((std::string(shaderDir) + "grass_blade.frag.spv").c_str(), moe::rhi::ShaderStage::kFragment)
                || !data->mGrassProgram.AddShader(data->mGrassVert)
                || !data->mGrassProgram.AddShader(data->mGrassFrag)) {
            std::fprintf(stderr, "grass: grass shader load failed\n");
            return false;
        }
        if (!data->mGroundVert.Load((std::string(shaderDir) + "grass_ground.vert.spv").c_str(), moe::rhi::ShaderStage::kVertex)
                || !data->mGroundFrag.Load((std::string(shaderDir) + "grass_ground.frag.spv").c_str(), moe::rhi::ShaderStage::kFragment)
                || !data->mGroundProgram.AddShader(data->mGroundVert)
                || !data->mGroundProgram.AddShader(data->mGroundFrag)) {
            std::fprintf(stderr, "grass: ground shader load failed\n");
            return false;
        }
        if (!data->mBlitVert.Load((std::string(shaderDir) + "grass_vfx.vert.spv").c_str(), moe::rhi::ShaderStage::kVertex)
                || !data->mBlitFrag.Load((std::string(shaderDir) + "grass_vfx.frag.spv").c_str(), moe::rhi::ShaderStage::kFragment)
                || !data->mBlitProgram.AddShader(data->mBlitVert)
                || !data->mBlitProgram.AddShader(data->mBlitFrag)) {
            std::fprintf(stderr, "grass: blit shader load failed\n");
            return false;
        }
        if (!data->mNoiseComp.Load((std::string(shaderDir) + "grass_noise.comp.spv").c_str(), moe::rhi::ShaderStage::kCompute)
                || !data->mNoiseProgram.AddShader(data->mNoiseComp)) {
            std::fprintf(stderr, "grass: noise shader load failed\n");
            return false;
        }

        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount)) {
            std::fprintf(stderr, "grass: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mSceneTarget = data->mRenderer.CreateRenderTarget(
                ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(),
                moe::rhi::Format::kR8G8B8A8Unorm, true);
        if (!data->mSceneTarget.IsValid()) {
            std::fprintf(stderr, "grass: scene target: %s\n", moe::Error::Get().c_str());
            return false;
        }

        moe::rhi::ImageCreateInfo noiseInfo{};
        noiseInfo.mType = moe::rhi::ImageType::k2D;
        noiseInfo.mWidth = kNoiseSize;
        noiseInfo.mHeight = kNoiseSize;
        noiseInfo.mFormat = moe::rhi::Format::kR8G8B8A8Unorm;
        noiseInfo.mUsage = moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled;
        if (!ctx.mDevice.CreateImage(noiseInfo, data->mNoiseImage)) {
            std::fprintf(stderr, "grass: noise image: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mMinFilter = moe::rhi::Filter::kLinear;
        samplerInfo.mMagFilter = moe::rhi::Filter::kLinear;
        if (!ctx.mDevice.CreateSampler(samplerInfo, data->mNoiseSampler)) {
            std::fprintf(stderr, "grass: noise sampler failed\n");
            return false;
        }
        moe::rhi::ComputePipelineState noiseState{};
        noiseState.mProgram = &data->mNoiseProgram;
        if (!ctx.mDevice.GetOrCreateComputePipeline(noiseState, data->mNoisePipeline)) {
            std::fprintf(stderr, "grass: noise pipeline: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::DescriptorSetLayout noiseLayout;
        if (!data->mNoisePipeline.GetDescriptorSetLayout(0, noiseLayout)
                || !ctx.mDevice.CreateDescriptorSet(noiseLayout, data->mNoiseSet)
                || !data->mNoiseSet.WriteImage(0, data->mNoiseImage, moe::rhi::DescriptorType::kStorageImage)) {
            std::fprintf(stderr, "grass: noise descriptor set failed\n");
            return false;
        }

        data->mSceneMvp = data->mRenderer.GetPushConstant(data->mSceneProgram, "mvp");
        data->mSceneModel = data->mRenderer.GetPushConstant(data->mSceneProgram, "model");
        data->mGrassMvp = data->mRenderer.GetPushConstant(data->mGrassProgram, "mvp");
        data->mGrassTime = data->mRenderer.GetPushConstant(data->mGrassProgram, "time");
        data->mGroundMvp = data->mRenderer.GetPushConstant(data->mGroundProgram, "mvp");
        data->mBlitDisplace = data->mRenderer.GetPushConstant(data->mBlitProgram, "displace");
        data->mBlitChromatic = data->mRenderer.GetPushConstant(data->mBlitProgram, "chromatic");
        data->mBlitVignette = data->mRenderer.GetPushConstant(data->mBlitProgram, "vignette");
        data->mBlitScanline = data->mRenderer.GetPushConstant(data->mBlitProgram, "scanline");
        if (data->mSceneMvp < 0 || data->mSceneModel < 0 || data->mGrassMvp < 0 || data->mGrassTime < 0
                || data->mGroundMvp < 0 || data->mBlitDisplace < 0 || data->mBlitChromatic < 0
                || data->mBlitVignette < 0 || data->mBlitScanline < 0) {
            std::fprintf(stderr, "grass: push constant names mismatch\n");
            return false;
        }

        // wire up the CRTP passes
        data->mScenePass.mTarget = data->mSceneTarget;
        data->mScenePass.mData = data;
        data->mBlitPass.mData = data;
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<PostfxData*>(userdata);
        data->mTime += 0.016f;

        const float clear[4] = {0.1f, 0.12f, 0.15f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        // noise update first: the scene pass samples it (grass wind), so it
        // must be in ShaderReadOnly layout before any draw. This is also the
        // first initialization of the image.
        moe::rhi::SyncInfo noiseSync{};
        noiseSync.mSrcStage = data->mNoiseLayout == moe::rhi::ImageLayout::kUndefined
                ? moe::rhi::PipelineStage::kTopOfPipe
                : moe::rhi::PipelineStage::kFragmentShader;
        noiseSync.mSrcAccess = data->mNoiseLayout == moe::rhi::ImageLayout::kUndefined
                ? moe::rhi::Access::kNone
                : moe::rhi::Access::kShaderRead;
        noiseSync.mDstStage = moe::rhi::PipelineStage::kComputeShader;
        noiseSync.mDstAccess = moe::rhi::Access::kShaderWrite;
        data->mRenderer.ImageBarrier(data->mNoiseImage, data->mNoiseLayout,
                moe::rhi::ImageLayout::kGeneral, noiseSync);
        data->mNoiseLayout = moe::rhi::ImageLayout::kGeneral;
        cmd.BindDescriptorSet(data->mNoisePipeline, data->mNoiseSet, 0);
        const float time = data->mTime;
        cmd.SetPushConstants(data->mNoisePipeline, 0, sizeof(float) * 4, &time);
        cmd.Dispatch(data->mNoisePipeline, kNoiseSize / 8, kNoiseSize / 8, 1);
        noiseSync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        noiseSync.mSrcAccess = moe::rhi::Access::kShaderWrite;
        noiseSync.mDstStage = moe::rhi::PipelineStage::kFragmentShader;
        noiseSync.mDstAccess = moe::rhi::Access::kShaderRead;
        data->mRenderer.ImageBarrier(data->mNoiseImage, moe::rhi::ImageLayout::kGeneral,
                moe::rhi::ImageLayout::kShaderReadOnly, noiseSync);
        data->mNoiseLayout = moe::rhi::ImageLayout::kShaderReadOnly;

        // pass 1: scene (ground + box + instanced grass) into the scene target
        data->mScenePass.Run(data->mRenderer);

        // pass 2: fullscreen composite sampling scene + noise
        data->mBlitPass.Run(data->mRenderer);

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<PostfxData*>(userdata);

        ImGui::Begin("grass demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::SliderFloat("displace", &data->mDisplace, 0.0f, 0.05f);
        ImGui::SliderFloat("chromatic", &data->mChromatic, 0.0f, 0.02f);
        ImGui::SliderFloat("vignette", &data->mVignette, 0.0f, 0.8f);
        ImGui::SliderFloat("scanline", &data->mScanline, 0.0f, 0.5f);
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<PostfxData*>(userdata);
        data->mNoiseSet.Destroy();
        data->mNoiseSampler.Destroy();
        data->mNoiseImage.Destroy();
        data->mInstanceBuffer.Destroy();
        data->mBoxTexture.Destroy();
        data->mGroundMesh.Destroy();
        data->mGrassMesh.Destroy();
        data->mBoxMesh.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    PostfxData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("grass demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "grass: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
