#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Pipeline.hpp>
#include <UI/Im3dDrawer.hpp>

#include <im3d.h>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
    constexpr uint32_t kLightCount = 256;
    constexpr uint32_t kTileSize = 16;
    constexpr uint32_t kMaxLightsPerTile = 128;

    struct Light {
        glm::vec3 mPosition;
        float mRadius;
        glm::vec3 mColor;
    };

    struct CullPushConstants {
        glm::mat4 mViewProj; // 64
        uint32_t mScreenWidth;
        uint32_t mScreenHeight;
        uint32_t mTileSize;
        uint32_t mMaxLightsPerTile;
        uint32_t mTileCountX;
        uint32_t mTileCount;
        uint32_t mLightCount;
        float mTanHalfFov;
    }; // 96

    struct ForwardPushConstants {
        glm::mat4 mViewProj; // 64
        glm::vec3 mCameraPos; // 12
        float mTileSize; // 4
        uint32_t mMaxLightsPerTile; // 4
        uint32_t mScreenWidth; // 4
        uint32_t mScreenHeight; // 4
        uint32_t mTileCountX; // 4
        float mPadding; // 4
    }; // 100

    // Forward+: a raw compute pass culls lights into per-tile lists between
    // Renderer passes, then a Renderer pass draws the scene with the tile
    // buffers bound as storage buffers.
    struct ForwardPlusData {
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::ProgramHandle mForwardProgram;
        moe::neo::UploadedMesh mMesh;
        moe::rhi::Buffer mLights;
        moe::rhi::Buffer mTileCounts;
        moe::rhi::Buffer mLightIndices;
        moe::rhi::Shader mCullComp;
        moe::rhi::ShaderProgram mCullProgram;
        moe::rhi::ComputePipeline mCullPipeline;
        moe::rhi::DescriptorSetLayout mSetLayout;
        moe::rhi::DescriptorSet mSet;
        moe::ui::Im3dDrawer* mIm3d{nullptr};
        std::vector<Light> mLightData; // CPU copy for the Im3d gizmos
        bool mShowGizmos{true};
        glm::vec3 mCameraPos{0.0f};
        glm::vec3 mForward{0.0f, 0.0f, -1.0f};
        glm::mat4 mViewProj{1.0f};
        CullPushConstants mCullPc{};
        ForwardPushConstants mForwardPc{};
        uint32_t mTilesX{0};
        uint32_t mTilesY{0};
        int32_t mPcViewProj{-1};
        int32_t mPcCameraPos{-1};
        int32_t mPcTileSize{-1};
        int32_t mPcMaxLights{-1};
        int32_t mPcScreenWidth{-1};
        int32_t mPcScreenHeight{-1};
        int32_t mPcTileCountX{-1};
    };

    void AppendBox(moe::neo::Mesh& mesh, const glm::vec3& center, const glm::vec3& half) {
        const glm::vec3 corners[8] = {
                center + glm::vec3(-half.x, -half.y, -half.z),
                center + glm::vec3(half.x, -half.y, -half.z),
                center + glm::vec3(half.x, half.y, -half.z),
                center + glm::vec3(-half.x, half.y, -half.z),
                center + glm::vec3(-half.x, -half.y, half.z),
                center + glm::vec3(half.x, -half.y, half.z),
                center + glm::vec3(half.x, half.y, half.z),
                center + glm::vec3(-half.x, half.y, half.z)};
        const glm::vec3 faceNormals[6] = {
                {0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
        const int faceCorners[6][4] = {
                {0, 1, 2, 3}, {5, 4, 7, 6}, {0, 4, 5, 1}, {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 5, 6, 2}};
        const glm::vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

        moe::neo::MeshPrimitive& prim = mesh.mPrimitives[0];
        for (int face = 0; face < 6; ++face) {
            for (int i = 0; i < 4; ++i) {
                const int corner = faceCorners[face][i];
                prim.mPositions.push_back(corners[corner]);
                prim.mNormals.push_back(faceNormals[face]);
                prim.mUv0.push_back(uvs[i]);
            }
            const uint32_t base = static_cast<uint32_t>(prim.mPositions.size()) - 4;
            prim.mIndices.push_back(base);
            prim.mIndices.push_back(base + 2);
            prim.mIndices.push_back(base + 1);
            prim.mIndices.push_back(base);
            prim.mIndices.push_back(base + 3);
            prim.mIndices.push_back(base + 2);
        }
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<ForwardPlusData*>(userdata);

        const uint32_t width = ctx.mSwapchain.GetWidth();
        const uint32_t height = ctx.mSwapchain.GetHeight();
        data->mTilesX = (width + kTileSize - 1) / kTileSize;
        data->mTilesY = (height + kTileSize - 1) / kTileSize;
        const uint32_t tileCount = data->mTilesX * data->mTilesY;

        // ---- scene: ground + a few boxes (world-space baked) ----
        moe::neo::Mesh mesh;
        mesh.mName = "forwardplus-scene";
        mesh.mPrimitives.emplace_back();
        AppendBox(mesh, glm::vec3(0.0f, -0.1f, 0.0f), glm::vec3(20.0f, 0.1f, 20.0f)); // ground
        AppendBox(mesh, glm::vec3(-6.0f, 1.5f, -4.0f), glm::vec3(1.5f, 1.5f, 1.5f));
        AppendBox(mesh, glm::vec3(5.0f, 1.0f, -6.0f), glm::vec3(1.0f, 1.0f, 1.0f));
        AppendBox(mesh, glm::vec3(3.0f, 3.0f, 4.0f), glm::vec3(2.0f, 3.0f, 2.0f));
        AppendBox(mesh, glm::vec3(-8.0f, 0.8f, 6.0f), glm::vec3(0.8f, 0.8f, 0.8f));
        AppendBox(mesh, glm::vec3(8.0f, 1.2f, 5.0f), glm::vec3(1.2f, 1.2f, 1.2f));

        if (!ctx.mTransfer.UploadMesh(mesh, data->mMesh)) {
            std::fprintf(stderr, "forwardplus: upload: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // ---- buffers: lights (CPU) + tile lists (GPU) ----
        moe::rhi::BufferCreateInfo bufferInfo{};
        bufferInfo.mSize = sizeof(Light) * kLightCount;
        bufferInfo.mUsage = moe::rhi::BufferUsage::kStorage;
        bufferInfo.mCpuVisible = true;
        if (!ctx.mDevice.CreateBuffer(bufferInfo, data->mLights)) {
            std::fprintf(stderr, "forwardplus: lights buffer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        bufferInfo.mCpuVisible = false;
        bufferInfo.mSize = sizeof(uint32_t) * tileCount;
        if (!ctx.mDevice.CreateBuffer(bufferInfo, data->mTileCounts)) {
            std::fprintf(stderr, "forwardplus: tile counts: %s\n", moe::Error::Get().c_str());
            return false;
        }
        bufferInfo.mSize = sizeof(uint32_t) * tileCount * kMaxLightsPerTile;
        if (!ctx.mDevice.CreateBuffer(bufferInfo, data->mLightIndices)) {
            std::fprintf(stderr, "forwardplus: light indices: %s\n", moe::Error::Get().c_str());
            return false;
        }

        {
            auto* lights = static_cast<Light*>(data->mLights.Map());
            if (!lights) {
                std::fprintf(stderr, "forwardplus: map lights failed\n");
                return false;
            }
            uint32_t state = 0x12345678u;
            const auto rnd = [&state]() {
                state = state * 1664525u + 1013904223u;
                return static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(1 << 24);
            };
            for (uint32_t i = 0; i < kLightCount; ++i) {
                Light light;
                light.mPosition = glm::vec3(rnd() * 36.0f - 18.0f, rnd() * 3.0f + 1.0f,
                        rnd() * 36.0f - 18.0f);
                light.mRadius = rnd() * 3.0f + 2.0f;
                const float t = rnd() * 0.6f + 0.4f;
                light.mColor = rnd() < 0.5f
                        ? glm::vec3(1.0f, 0.85f, 0.7f) * t
                        : glm::vec3(0.5f, 0.7f, 1.0f) * t;
                lights[i] = light;
                data->mLightData.push_back(light);
            }
            data->mLights.Unmap();
        }

        // ---- cull compute (raw: runs between Renderer passes) ----
        if (!data->mCullComp.Load(MOE_SOURCE_DIR "/shaders/examples/forwardplus/forwardplus_cull.comp.spv",
                    moe::rhi::ShaderStage::kCompute)
                || !data->mCullProgram.AddShader(data->mCullComp)) {
            std::fprintf(stderr, "forwardplus: cull shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::ComputePipelineState cullState{};
        cullState.mProgram = &data->mCullProgram;
        if (!ctx.mDevice.GetOrCreateComputePipeline(cullState, data->mCullPipeline)
                || !data->mCullPipeline.GetDescriptorSetLayout(0, data->mSetLayout)
                || !ctx.mDevice.CreateDescriptorSet(data->mSetLayout, data->mSet)
                || !data->mSet.WriteBuffer(0, data->mLights)
                || !data->mSet.WriteBuffer(1, data->mTileCounts)
                || !data->mSet.WriteBuffer(2, data->mLightIndices)) {
            std::fprintf(stderr, "forwardplus: cull pipeline/set: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // ---- forward program (content layer) + renderer ----
        data->mForwardProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/forwardplus/forwardplus.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/forwardplus/forwardplus.frag.spv");
        if (!data->mForwardProgram.IsValid()) {
            std::fprintf(stderr, "forwardplus: forward shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache, width, height,
                    ctx.mSampleCount, ctx.mTransfer)) {
            std::fprintf(stderr, "forwardplus: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }

        const moe::rhi::ShaderProgram* forward = ctx.mAssets.GetProgram(data->mForwardProgram);
        data->mPcViewProj = data->mRenderer.GetPushConstant(*forward, "viewProj");
        data->mPcCameraPos = data->mRenderer.GetPushConstant(*forward, "cameraPos");
        data->mPcTileSize = data->mRenderer.GetPushConstant(*forward, "tileSize");
        data->mPcMaxLights = data->mRenderer.GetPushConstant(*forward, "maxLightsPerTile");
        data->mPcScreenWidth = data->mRenderer.GetPushConstant(*forward, "screenWidth");
        data->mPcScreenHeight = data->mRenderer.GetPushConstant(*forward, "screenHeight");
        data->mPcTileCountX = data->mRenderer.GetPushConstant(*forward, "tileCountX");
        if (data->mPcViewProj < 0 || data->mPcCameraPos < 0 || data->mPcTileSize < 0
                || data->mPcMaxLights < 0 || data->mPcScreenWidth < 0
                || data->mPcScreenHeight < 0 || data->mPcTileCountX < 0) {
            std::fprintf(stderr, "forwardplus: push constant names mismatch\n");
            return false;
        }

        data->mIm3d = &ctx.mIm3d;
        ctx.mInput.BindAction(0, static_cast<int32_t>(moe::neo::KeyCode::kSpace));

        data->mCullPc.mScreenWidth = width;
        data->mCullPc.mScreenHeight = height;
        data->mCullPc.mTileSize = kTileSize;
        data->mCullPc.mMaxLightsPerTile = kMaxLightsPerTile;
        data->mCullPc.mTileCountX = data->mTilesX;
        data->mCullPc.mTileCount = tileCount;
        data->mCullPc.mLightCount = kLightCount;
        data->mCullPc.mTanHalfFov = std::tan(glm::radians(60.0f) * 0.5f);
        data->mForwardPc.mTileSize = static_cast<float>(kTileSize);
        data->mForwardPc.mMaxLightsPerTile = kMaxLightsPerTile;
        data->mForwardPc.mScreenWidth = width;
        data->mForwardPc.mScreenHeight = height;
        data->mForwardPc.mTileCountX = data->mTilesX;
        return true;
    }

    void UpdateCamera(ForwardPlusData& data, uint32_t width, uint32_t height) {
        static float angle = 0.0f;
        angle += 0.0025f;
        const float radius = 18.0f;
        const glm::vec3 target(0.0f, 0.0f, 0.0f);
        data.mCameraPos = target + glm::vec3(std::cos(angle), 0.0f, std::sin(angle)) * radius;
        data.mCameraPos.y = 12.0f; // overview of the scene
        data.mForward = glm::normalize(target - data.mCameraPos);
        const glm::mat4 view = glm::lookAt(data.mCameraPos, target, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                static_cast<float>(width) / static_cast<float>(height), 0.1f, 100.0f);
        proj[1][1] *= -1; // Vulkan NDC: flip Y
        data.mViewProj = proj * view;
    }

    void DrawIm3d(void* userdata, examples::AppContext& ctx, float deltaSeconds) {
        auto* data = static_cast<ForwardPlusData*>(userdata);
        if (data->mIm3d == nullptr || !data->mIm3d->IsActive()) {
            return;
        }
        const uint32_t width = ctx.mSwapchain.GetWidth();
        const uint32_t height = ctx.mSwapchain.GetHeight();

        UpdateCamera(*data, width, height); // the scene camera must keep moving

        // space toggles the gizmos (input validation: action binding + edges)
        if (ctx.mInput.IsActionJustPressed(0)) {
            data->mShowGizmos = !data->mShowGizmos;
        }

        // The Im3d frame lifecycle (NewFrame/EndFrame) runs unconditionally:
        // skipping NewFrame would leave the previous frame's primitives stale,
        // so the drawer would re-upload frozen gizmos instead of none.
        Im3d::AppData& appData = Im3d::GetAppData();
        appData.m_deltaTime = deltaSeconds;
        appData.m_viewportSize = Im3d::Vec2(static_cast<float>(width), static_cast<float>(height));
        appData.m_projOrtho = false;
        appData.m_viewOrigin = Im3d::Vec3(data->mCameraPos.x, data->mCameraPos.y, data->mCameraPos.z);
        appData.m_viewDirection = Im3d::Vec3(data->mForward.x, data->mForward.y, data->mForward.z);
        appData.m_worldUp = Im3d::Vec3(0.0f, 1.0f, 0.0f);
        const float fovXDeg = 60.0f * (static_cast<float>(width) / static_cast<float>(height));
        appData.m_projScaleY = std::tan(glm::radians(fovXDeg));
        appData.m_keyDown[Im3d::Mouse_Left] = false;

        Im3d::NewFrame();

        // gizmos: light points + a ground grid + the first light's box
        if (data->mShowGizmos) {
            for (const auto& light : data->mLightData) {
                const Im3d::Color color(
                        static_cast<uint32_t>(glm::clamp(light.mColor.r, 0.0f, 1.0f) * 255.0f) << 24
                        | static_cast<uint32_t>(glm::clamp(light.mColor.g, 0.0f, 1.0f) * 255.0f) << 16
                        | static_cast<uint32_t>(glm::clamp(light.mColor.b, 0.0f, 1.0f) * 255.0f) << 8
                        | 0xFFu);
                Im3d::DrawPoint(Im3d::Vec3(light.mPosition.x, light.mPosition.y, light.mPosition.z),
                        8.0f, color);
            }

            // ground grid (lines, exercises the geometry-shader line pipeline)
            Im3d::PushColor(Im3d::Color(0x666666ff));
            const float gridHalf = 18.0f;
            for (int i = -9; i <= 9; ++i) {
                const float x = i * 2.0f;
                Im3d::DrawLine(Im3d::Vec3(x, 0.0f, -gridHalf), Im3d::Vec3(x, 0.0f, gridHalf), 1.0f, Im3d::Color(0x666666ff));
                Im3d::DrawLine(Im3d::Vec3(-gridHalf, 0.0f, x), Im3d::Vec3(gridHalf, 0.0f, x), 1.0f, Im3d::Color(0x666666ff));
            }
            Im3d::PopColor();

            // the first light as a box (triangles pipeline)
            const Light& first = data->mLightData[0];
            Im3d::PushColor(Im3d::Color(0xffd700ff));
            const float s = 0.35f;
            Im3d::DrawAlignedBox(
                    Im3d::Vec3(first.mPosition.x - s, first.mPosition.y - s, first.mPosition.z - s),
                    Im3d::Vec3(first.mPosition.x + s, first.mPosition.y + s, first.mPosition.z + s));
            Im3d::PopColor();
        }

        Im3d::EndFrame();

        data->mIm3d->mViewProj = data->mViewProj;
        data->mIm3d->mViewport = glm::vec2(static_cast<float>(width), static_cast<float>(height));
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<ForwardPlusData*>(userdata);

        data->mCullPc.mViewProj = data->mViewProj;
        data->mForwardPc.mViewProj = data->mViewProj;
        data->mForwardPc.mCameraPos = data->mCameraPos;

        const float clear[4] = {0.02f, 0.02f, 0.03f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        // Im3d vertex upload is a transfer: must be outside a render pass
        if (data->mIm3d != nullptr && data->mIm3d->IsActive()) {
            data->mIm3d->UploadVertices(cmd);
        }

        // light culling (raw compute between passes)
        cmd.BindDescriptorSet(data->mCullPipeline, data->mSet, 0);
        cmd.SetPushConstants(data->mCullPipeline, 0, sizeof(CullPushConstants), &data->mCullPc);
        cmd.Dispatch(data->mCullPipeline, data->mTilesX, data->mTilesY, 1);

        // compute writes -> fragment reads (tile lists)
        moe::rhi::SyncInfo sync{};
        sync.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        sync.mSrcAccess = moe::rhi::Access::kShaderWrite;
        sync.mDstStage = moe::rhi::PipelineStage::kFragmentShader;
        sync.mDstAccess = moe::rhi::Access::kShaderRead;
        data->mRenderer.MemoryBarrier(sync);

        // forward pass: scene + Im3d gizmos
        const moe::neo::PassDesc pass{"forward+", {}, {}};
        data->mRenderer.Execute(pass, [&](moe::neo::PassContext& context) {
            context.BindBuffer(0, data->mLights);
            context.BindBuffer(1, data->mTileCounts);
            context.BindBuffer(2, data->mLightIndices);
            context.SetPushConstant(data->mPcViewProj, &data->mForwardPc.mViewProj,
                    sizeof(glm::mat4));
            context.SetPushConstant(data->mPcCameraPos, &data->mForwardPc.mCameraPos,
                    sizeof(glm::vec3));
            context.SetPushConstant(data->mPcTileSize, &data->mForwardPc.mTileSize, sizeof(float));
            context.SetPushConstant(data->mPcMaxLights, &data->mForwardPc.mMaxLightsPerTile,
                    sizeof(uint32_t));
            context.SetPushConstant(data->mPcScreenWidth, &data->mForwardPc.mScreenWidth,
                    sizeof(uint32_t));
            context.SetPushConstant(data->mPcScreenHeight, &data->mForwardPc.mScreenHeight,
                    sizeof(uint32_t));
            context.SetPushConstant(data->mPcTileCountX, &data->mForwardPc.mTileCountX,
                    sizeof(uint32_t));
            context.Draw(data->mMesh, *ctx.mAssets.GetProgram(data->mForwardProgram));
            // raw overlay last: it binds its own pipelines (the renderer's
            // pipeline tracking resets at the next pass)
            if (data->mIm3d != nullptr && data->mIm3d->IsActive()) {
                data->mIm3d->Record(cmd);
            }
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<ForwardPlusData*>(userdata);
        data->mSet.Destroy();
        data->mLightIndices.Destroy();
        data->mTileCounts.Destroy();
        data->mLights.Destroy();
        data->mMesh.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    ForwardPlusData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mDrawIm3d = DrawIm3d;
    callbacks.mPostRender = PostRender;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("forward+ demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "forwardplus: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
