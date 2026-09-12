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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {
    // Mœbius-style toon demo with hand-drawn "boiling" ink:
    //   pass 1: scene (toon / toon_spec per material) -> scene target
    //   pass 2: normal+depth prepass -> 1x target (outline source)
    //   pass 3: screen-space ink: silhouettes/creases from the prepass,
    //           dilated to a constant pixel width and wobbled by a seamless
    //           noise field (animated on twos), over the resolved scene
    struct ToonData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::Model mModel;
        moe::neo::ProgramHandle mToonProgram;
        moe::neo::ProgramHandle mToonSpecProgram;
        moe::neo::ProgramHandle mNormalProgram;
        moe::neo::ProgramHandle mCompositeProgram;
        moe::neo::TextureHandle mNoiseAtlas;
        moe::neo::RenderTargetHandle mSceneTarget;
        moe::neo::RenderTargetHandle mNormalTarget;
        int32_t mCompositeResolution{-1};
        int32_t mCompositeAmplitude{-1};
        int32_t mCompositeNoiseFrame{-1};
        int32_t mCompositeNoiseScale{-1};
        int32_t mCompositeLineWidth{-1};
        int32_t mCompositeNormalThreshold{-1};
        int32_t mCompositeDepthThreshold{-1};
        int32_t mCompositeInkColor{-1};
        float mTime{0.0f};
        float mNoiseFrame{0.0f};
        float mSteps{3.0f};
        float mRim{0.25f};
        float mSpecStrength{0.15f};
        float mInkWidth{2.5f};
        float mJitterAmplitude{1.0f};
        float mNoiseScale{4.0f};
        float mInkRate{10.0f};
        float mNormalThreshold{0.35f};
        float mDepthThreshold{0.08f};
        float mInkColor[4] = {0.10f, 0.09f, 0.12f, 1.0f};
        bool mOutlineEnabled{true};
    };

    constexpr const char* kMaterialNames[] = {"sphere", "torus", "box", "ground"};

    // ---- procedural geometry (normals required: toon shading + outline) ----

    moe::neo::Mesh MakeSphere(float radius, uint32_t segments, uint32_t rings) {
        moe::neo::Mesh mesh;
        mesh.mName = "sphere";
        moe::neo::MeshPrimitive prim;
        for (uint32_t y = 0; y <= rings; ++y) {
            const float v = static_cast<float>(y) / static_cast<float>(rings);
            const float phi = v * glm::pi<float>();
            for (uint32_t x = 0; x <= segments; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(segments);
                const float theta = u * glm::two_pi<float>();
                const glm::vec3 normal(std::cos(theta) * std::sin(phi), std::cos(phi),
                        std::sin(theta) * std::sin(phi));
                prim.mPositions.push_back(normal * radius);
                prim.mNormals.push_back(normal);
                prim.mUv0.push_back({u, v});
            }
        }
        for (uint32_t y = 0; y < rings; ++y) {
            for (uint32_t x = 0; x < segments; ++x) {
                const uint32_t a = y * (segments + 1) + x;
                const uint32_t b = a + 1;
                const uint32_t c = a + segments + 1;
                const uint32_t d = c + 1;
                prim.mIndices.push_back(a);
                prim.mIndices.push_back(b);
                prim.mIndices.push_back(c);
                prim.mIndices.push_back(b);
                prim.mIndices.push_back(d);
                prim.mIndices.push_back(c);
            }
        }
        mesh.mPrimitives.push_back(std::move(prim));
        return mesh;
    }

    moe::neo::Mesh MakeTorus(float major, float minor, uint32_t majorSegments,
            uint32_t minorSegments) {
        moe::neo::Mesh mesh;
        mesh.mName = "torus";
        moe::neo::MeshPrimitive prim;
        for (uint32_t i = 0; i <= majorSegments; ++i) {
            const float a = static_cast<float>(i) / static_cast<float>(majorSegments)
                    * glm::two_pi<float>();
            const glm::vec3 center(std::cos(a) * major, 0.0f, std::sin(a) * major);
            for (uint32_t j = 0; j <= minorSegments; ++j) {
                const float b = static_cast<float>(j) / static_cast<float>(minorSegments)
                        * glm::two_pi<float>();
                const glm::vec3 normal(std::cos(a) * std::cos(b), std::sin(b),
                        std::sin(a) * std::cos(b));
                prim.mPositions.push_back(center + normal * minor);
                prim.mNormals.push_back(normal);
                prim.mUv0.push_back({static_cast<float>(i) / static_cast<float>(majorSegments),
                        static_cast<float>(j) / static_cast<float>(minorSegments)});
            }
        }
        for (uint32_t i = 0; i < majorSegments; ++i) {
            for (uint32_t j = 0; j < minorSegments; ++j) {
                const uint32_t a = i * (minorSegments + 1) + j;
                const uint32_t b = a + 1;
                const uint32_t c = a + minorSegments + 1;
                const uint32_t d = c + 1;
                prim.mIndices.push_back(a);
                prim.mIndices.push_back(b);
                prim.mIndices.push_back(c);
                prim.mIndices.push_back(b);
                prim.mIndices.push_back(d);
                prim.mIndices.push_back(c);
            }
        }
        mesh.mPrimitives.push_back(std::move(prim));
        return mesh;
    }

    moe::neo::Mesh MakeBox(float halfSize) {
        moe::neo::Mesh mesh;
        mesh.mName = "box";
        moe::neo::MeshPrimitive prim;
        const float h = halfSize;
        const glm::vec3 corners[8] = {
                {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
                {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}};
        const glm::vec3 faceNormals[6] = {
                {0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
        const int faceCorners[6][4] = {
                {0, 1, 2, 3}, {5, 4, 7, 6}, {0, 4, 5, 1}, {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 5, 6, 2}};
        const glm::vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
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

    moe::neo::Mesh MakeGround(float halfSize) {
        moe::neo::Mesh mesh;
        mesh.mName = "ground";
        moe::neo::MeshPrimitive prim;
        const glm::vec3 corners[4] = {
                {-halfSize, 0.0f, -halfSize}, {halfSize, 0.0f, -halfSize},
                {halfSize, 0.0f, halfSize}, {-halfSize, 0.0f, halfSize}};
        const glm::vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (int i = 0; i < 4; ++i) {
            prim.mPositions.push_back(corners[i]);
            prim.mNormals.push_back({0.0f, 1.0f, 0.0f});
            prim.mUv0.push_back(uvs[i]);
        }
        prim.mIndices = {0, 2, 1, 0, 3, 2};
        mesh.mPrimitives.push_back(std::move(prim));
        return mesh;
    }

    // 2x2 atlas of 4 temporally-coherent slices of a seamless 3D value noise
    // (rg = jitter offset). x/y wrap, so the screen-space tiling has no seams,
    // and z advances smoothly per tile, so the "boiling" reads as an evolving
    // wobble instead of four unrelated patterns.
    moe::neo::Texture MakeNoiseAtlas(uint32_t tileSize, uint32_t gridSize) {
        constexpr uint32_t kZGrid = 5;  // slices along z (4 frames + interpolation)
        constexpr float kZStep = 0.7f;  // how far apart consecutive frames sit

        moe::neo::Texture texture;
        texture.mName = "toon_noise_atlas";
        texture.mWidth = tileSize * 2;
        texture.mHeight = tileSize * 2;
        texture.mChannels = 4;
        texture.mSrgb = false;
        texture.mData.resize(static_cast<size_t>(texture.mWidth) * texture.mHeight * 4);

        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<glm::vec2> grid(static_cast<size_t>(kZGrid) * gridSize * gridSize);
        for (glm::vec2& value : grid) {
            value = {dist(rng), dist(rng)};
        }
        const auto sampleGrid = [&](uint32_t x, uint32_t y, uint32_t z) {
            return grid[(static_cast<size_t>(z) * gridSize + (y % gridSize)) * gridSize
                    + (x % gridSize)];
        };
        const auto quintic = [](float t) {
            return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        };

        for (uint32_t tile = 0; tile < 4; ++tile) {
            const float fz = static_cast<float>(tile) * kZStep;
            const uint32_t z0 = static_cast<uint32_t>(fz);
            const float tz = quintic(fz - static_cast<float>(z0));
            const uint32_t tileX = (tile % 2) * tileSize;
            const uint32_t tileY = (tile / 2) * tileSize;
            for (uint32_t y = 0; y < tileSize; ++y) {
                const float fy = static_cast<float>(y) / static_cast<float>(tileSize) * gridSize;
                const uint32_t y0 = static_cast<uint32_t>(fy);
                const float ty = quintic(fy - static_cast<float>(y0));
                for (uint32_t x = 0; x < tileSize; ++x) {
                    const float fx = static_cast<float>(x) / static_cast<float>(tileSize) * gridSize;
                    const uint32_t x0 = static_cast<uint32_t>(fx);
                    const float tx = quintic(fx - static_cast<float>(x0));
                    glm::vec2 value(0.0f);
                    for (uint32_t dz = 0; dz < 2; ++dz) {
                        const glm::vec2 v00 = sampleGrid(x0, y0, z0 + dz);
                        const glm::vec2 v10 = sampleGrid(x0 + 1, y0, z0 + dz);
                        const glm::vec2 v01 = sampleGrid(x0, y0 + 1, z0 + dz);
                        const glm::vec2 v11 = sampleGrid(x0 + 1, y0 + 1, z0 + dz);
                        const glm::vec2 slice = glm::mix(glm::mix(v00, v10, tx),
                                glm::mix(v01, v11, tx), ty);
                        value = dz == 0 ? slice : glm::mix(value, slice, tz);
                    }
                    const size_t i = (static_cast<size_t>(tileY + y) * texture.mWidth
                            + (tileX + x)) * 4;
                    texture.mData[i + 0] = static_cast<uint8_t>(
                            glm::clamp(value.x, 0.0f, 1.0f) * 255.0f);
                    texture.mData[i + 1] = static_cast<uint8_t>(
                            glm::clamp(value.y, 0.0f, 1.0f) * 255.0f);
                    texture.mData[i + 2] = 128;
                    texture.mData[i + 3] = 255;
                }
            }
        }
        return texture;
    }

    uint32_t AddNode(moe::neo::Scene& scene, uint32_t meshIndex, const glm::vec3& translation) {
        moe::neo::Node node;
        node.mName = scene.mMeshes[meshIndex].mName;
        node.mTranslation = translation;
        node.mMeshes.push_back(meshIndex);
        const uint32_t index = static_cast<uint32_t>(scene.mGraph.mNodes.size());
        scene.mGraph.mNodes.push_back(std::move(node));
        scene.mGraph.mRootNodes.push_back(index);
        return index;
    }

    moe::neo::Material MakeMaterial(const char* name, const glm::vec4& baseColor) {
        moe::neo::Material material;
        material.mName = name;
        material.mBaseColor = baseColor;
        return material;
    }

    void ApplyMaterialParams(ToonData& data) {
        for (const char* name : kMaterialNames) {
            data.mModel.SetMaterialParam(name, "steps", data.mSteps);
            data.mModel.SetMaterialParam(name, "rimStrength", data.mRim);
            data.mModel.SetMaterialParam(name, "specStrength", data.mSpecStrength);
        }
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<ToonData*>(userdata);

        // procedural scene: one mesh + material + node per shape
        moe::neo::Scene scene;
        scene.mName = "toon";
        scene.mMeshes.push_back(MakeSphere(0.9f, 48, 32));
        scene.mMeshes.push_back(MakeTorus(0.75f, 0.28f, 64, 32));
        scene.mMeshes.push_back(MakeBox(0.7f));
        scene.mMeshes.push_back(MakeGround(8.0f));
        scene.mMaterials.push_back(MakeMaterial("sphere", {0.92f, 0.36f, 0.32f, 1.0f}));
        scene.mMaterials.push_back(MakeMaterial("torus", {0.30f, 0.55f, 0.85f, 1.0f}));
        scene.mMaterials.push_back(MakeMaterial("box", {0.95f, 0.78f, 0.30f, 1.0f}));
        scene.mMaterials.push_back(MakeMaterial("ground", {0.82f, 0.80f, 0.74f, 1.0f}));
        for (uint32_t i = 0; i < scene.mMeshes.size(); ++i) {
            scene.mMeshes[i].mPrimitives[0].mMaterialIndex = static_cast<int32_t>(i);
        }
        AddNode(scene, 0, {-2.6f, 1.0f, 0.0f});
        AddNode(scene, 1, {0.0f, 1.15f, 0.0f});
        AddNode(scene, 2, {2.6f, 0.7f, 0.0f});
        AddNode(scene, 3, {0.0f, 0.0f, 0.0f});

        data->mModel = ctx.mAssets.UploadScene(scene);
        if (!data->mModel.IsValid()) {
            std::fprintf(stderr, "toon: scene upload: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const char* shaderDir = MOE_SOURCE_DIR "/shaders/examples/toon/";
        data->mToonProgram = ctx.mAssets.LoadGraphicsProgram(
                (std::string(shaderDir) + "toon.vert.spv").c_str(),
                (std::string(shaderDir) + "toon.frag.spv").c_str());
        data->mToonSpecProgram = ctx.mAssets.LoadGraphicsProgram(
                (std::string(shaderDir) + "toon_spec.vert.spv").c_str(),
                (std::string(shaderDir) + "toon_spec.frag.spv").c_str());
        data->mNormalProgram = ctx.mAssets.LoadGraphicsProgram(
                (std::string(shaderDir) + "toon_normal.vert.spv").c_str(),
                (std::string(shaderDir) + "toon_normal.frag.spv").c_str());
        data->mCompositeProgram = ctx.mAssets.LoadGraphicsProgram(
                (std::string(shaderDir) + "toon_composite.vert.spv").c_str(),
                (std::string(shaderDir) + "toon_composite.frag.spv").c_str());
        if (!data->mToonProgram.IsValid() || !data->mToonSpecProgram.IsValid()
                || !data->mNormalProgram.IsValid() || !data->mCompositeProgram.IsValid()) {
            std::fprintf(stderr, "toon: shader load: %s\n", moe::Error::Get().c_str());
            return false;
        }

        data->mNoiseAtlas = ctx.mAssets.UploadTexture(MakeNoiseAtlas(256, 8));
        if (!data->mNoiseAtlas.IsValid()) {
            std::fprintf(stderr, "toon: noise atlas: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // per-material program: the torus gets the specular variant
        data->mModel.SetMaterialProgram("torus", data->mToonSpecProgram);
        ApplyMaterialParams(*data);

        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                    ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount, ctx.mTransfer)) {
            std::fprintf(stderr, "toon: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mSceneTarget = data->mRenderer.CreateRenderTarget(
                ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(),
                moe::rhi::Format::kR8G8B8A8Unorm, true);
        // outline source: single-sample so silhouettes stay crisp
        data->mNormalTarget = data->mRenderer.CreateRenderTarget(
                ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(),
                moe::rhi::Format::kR16G16B16A16Float, true, 1);
        if (!data->mSceneTarget.IsValid() || !data->mNormalTarget.IsValid()) {
            std::fprintf(stderr, "toon: render targets: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const moe::rhi::ShaderProgram* composite = ctx.mAssets.GetProgram(data->mCompositeProgram);
        data->mCompositeResolution = data->mRenderer.GetPushConstant(*composite, "resolution");
        data->mCompositeAmplitude = data->mRenderer.GetPushConstant(*composite, "amplitude");
        data->mCompositeNoiseFrame = data->mRenderer.GetPushConstant(*composite, "noiseFrame");
        data->mCompositeNoiseScale = data->mRenderer.GetPushConstant(*composite, "noiseScale");
        data->mCompositeLineWidth = data->mRenderer.GetPushConstant(*composite, "lineWidth");
        data->mCompositeNormalThreshold = data->mRenderer.GetPushConstant(*composite, "normalThreshold");
        data->mCompositeDepthThreshold = data->mRenderer.GetPushConstant(*composite, "depthThreshold");
        data->mCompositeInkColor = data->mRenderer.GetPushConstant(*composite, "inkColor");
        if (data->mCompositeResolution < 0 || data->mCompositeAmplitude < 0
                || data->mCompositeNoiseFrame < 0 || data->mCompositeNoiseScale < 0
                || data->mCompositeLineWidth < 0 || data->mCompositeNormalThreshold < 0
                || data->mCompositeDepthThreshold < 0 || data->mCompositeInkColor < 0) {
            std::fprintf(stderr, "toon: composite push constant names mismatch\n");
            return false;
        }
        return true;
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<ToonData*>(userdata);
        data->mTime += 0.016f;
        // "on twos": noise frames per second, cycling through the 4 tiles
        data->mNoiseFrame = std::fmod(std::floor(data->mTime * data->mInkRate), 4.0f);

        const float clear[4] = {0.90f, 0.88f, 0.84f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        const float angle = data->mTime * 0.25f;
        moe::neo::Camera camera;
        camera.mView = glm::lookAt(glm::vec3(6.0f * std::sin(angle), 3.4f, 6.0f * std::cos(angle)),
                glm::vec3(0.0f, 0.9f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        camera.mProj = glm::perspective(glm::radians(50.0f),
                static_cast<float>(data->mFrame.GetWidth())
                        / static_cast<float>(data->mFrame.GetHeight()),
                0.1f, 100.0f);
        camera.mProj[1][1] *= -1.0f;

        const glm::mat4 identity(1.0f);

        // pass 1: scene (toon / toon_spec per material)
        const moe::neo::PassDesc scenePass{"toon scene",
                moe::neo::ColorAttachment(data->mSceneTarget), {}};
        data->mRenderer.Execute(scenePass, [&](moe::neo::PassContext& context) {
            context.SetCamera(camera);
            context.DrawModel(data->mModel, data->mToonProgram, identity);
        });

        // pass 2: normal+depth prepass for the screen-space ink (1x target,
        // its own depth so only the visible surface is recorded)
        const moe::neo::PassDesc normalPass{"toon normal",
                moe::neo::ColorAttachment(data->mNormalTarget), {}};
        data->mRenderer.Execute(normalPass, [&](moe::neo::PassContext& context) {
            context.SetCamera(camera);
            context.DrawModelForced(data->mModel, data->mNormalProgram, identity);
        });

        // pass 3: ink: silhouette/crease detection over the scene
        const moe::neo::PassDesc compositePass{"toon ink", {}, {}};
        data->mRenderer.Execute(compositePass, [&](moe::neo::PassContext& context) {
            moe::neo::UploadedTexture* noise = ctx.mAssets.GetTexture(data->mNoiseAtlas);
            moe::neo::RenderTarget* scene = data->mRenderer.GetRenderTarget(data->mSceneTarget);
            moe::neo::RenderTarget* normal = data->mRenderer.GetRenderTarget(data->mNormalTarget);
            context.BindImage(0, *scene->mImage);
            context.BindSampler(1, noise->mSampler);
            context.BindImage(2, *normal->mImage);
            context.BindSampler(3, noise->mSampler);
            context.BindImage(4, noise->mImage);
            context.BindSampler(5, noise->mSampler);

            const float resolution[2] = {
                    static_cast<float>(data->mFrame.GetWidth()),
                    static_cast<float>(data->mFrame.GetHeight())};
            const float inkColor[4] = {
                    data->mInkColor[0], data->mInkColor[1], data->mInkColor[2],
                    data->mOutlineEnabled ? data->mInkColor[3] : 0.0f};
            context.SetPushConstant(data->mCompositeResolution, resolution, sizeof(resolution));
            context.SetPushConstant(data->mCompositeAmplitude, &data->mJitterAmplitude,
                    sizeof(data->mJitterAmplitude));
            context.SetPushConstant(data->mCompositeNoiseFrame, &data->mNoiseFrame,
                    sizeof(data->mNoiseFrame));
            context.SetPushConstant(data->mCompositeNoiseScale, &data->mNoiseScale,
                    sizeof(data->mNoiseScale));
            context.SetPushConstant(data->mCompositeLineWidth, &data->mInkWidth,
                    sizeof(data->mInkWidth));
            context.SetPushConstant(data->mCompositeNormalThreshold, &data->mNormalThreshold,
                    sizeof(data->mNormalThreshold));
            context.SetPushConstant(data->mCompositeDepthThreshold, &data->mDepthThreshold,
                    sizeof(data->mDepthThreshold));
            context.SetPushConstant(data->mCompositeInkColor, inkColor, sizeof(inkColor));
            context.DrawFullscreen(*ctx.mAssets.GetProgram(data->mCompositeProgram));
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<ToonData*>(userdata);
        ImGui::Begin("toon demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        bool changed = false;
        changed |= ImGui::SliderFloat("steps", &data->mSteps, 1.0f, 6.0f);
        changed |= ImGui::SliderFloat("rim", &data->mRim, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("spec", &data->mSpecStrength, 0.0f, 0.5f);
        if (changed) {
            ApplyMaterialParams(*data);
        }
        ImGui::Separator();
        ImGui::Checkbox("ink", &data->mOutlineEnabled);
        ImGui::ColorEdit4("ink color", data->mInkColor);
        ImGui::SliderFloat("ink width (px)", &data->mInkWidth, 0.5f, 6.0f);
        ImGui::SliderFloat("ink jitter (px)", &data->mJitterAmplitude, 0.0f, 4.0f);
        ImGui::SliderFloat("ink scale", &data->mNoiseScale, 1.0f, 12.0f);
        ImGui::SliderFloat("ink rate (Hz)", &data->mInkRate, 0.0f, 24.0f);
        ImGui::SliderFloat("crease threshold", &data->mNormalThreshold, 0.05f, 0.9f);
        ImGui::SliderFloat("depth threshold", &data->mDepthThreshold, 0.01f, 0.5f);
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<ToonData*>(userdata);
        data->mRenderer.DestroyRenderTarget(data->mNormalTarget);
        data->mRenderer.DestroyRenderTarget(data->mSceneTarget);
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    ToonData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("Mœbius toon demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "toon: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
