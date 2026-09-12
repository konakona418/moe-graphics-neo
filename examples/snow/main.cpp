// Snow/sand terrain demo: an interactive height field you walk on with the
// mouse-captured FPS camera. The height field lives in a GPU storage buffer;
// two compute passes update it every frame (brush sink under the player, then
// normal/color rebuild writing the vertex buffer directly), and a 17 KB
// readback keeps the CPU height copy the camera follows. Exercises the
// dynamic-GPU-data path: compute-writable vertex buffers + precise barriers.

#include <examples/common/App.hpp>

#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>
#include <Neo/SwapchainImage.hpp>
#include <Neo/TransferManager.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Pipeline.hpp>
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
#include <string>

namespace {
    constexpr uint32_t kGridSize = 64;  // 64x64 cells -> 65x65 vertices
    constexpr uint32_t kVerticesPerSide = kGridSize + 1;
    constexpr uint32_t kVertexCount = kVerticesPerSide * kVerticesPerSide;
    constexpr float kCellSize = 0.5f;
    constexpr float kTerrainSize = kGridSize * kCellSize;

    // interleaved vertex layout written by snow_build.comp (slang std430:
    // float3 @0, float3 @12, rgba8 @24, tightly packed 28-byte stride)
    constexpr uint32_t kVertexStride = 28;
    constexpr uint32_t kPositionOffset = 0;
    constexpr uint32_t kNormalOffset = 12;
    constexpr uint32_t kColorOffset = 24;

    constexpr uint32_t kActionMoveForward = 0;
    constexpr uint32_t kActionMoveBack = 1;
    constexpr uint32_t kActionMoveLeft = 2;
    constexpr uint32_t kActionMoveRight = 3;

    struct BrushPushConstants {
        glm::vec4 mParams; // xy = center (grid coords), z = radius, w = amount
    };

    struct BuildPushConstants {
        glm::vec4 mParams; // x = 1.0 snow / 0.0 sand
    };

    struct SnowData {
        // GPU-owned terrain state. The vertex/index buffers double as the
        // Renderer's UploadedMesh view (the build compute pass writes the
        // vertices directly, so they are raw storage buffers first).
        moe::rhi::Buffer mHeightBuffer;  // float[kVertexCount], storage|transfer
        moe::neo::UploadedMesh mMesh;    // vertex (storage|vertex) + index
        moe::rhi::Buffer mHeightReadback; // cpu-visible copy target
        moe::rhi::Shader mBrushComp;
        moe::rhi::Shader mBuildComp;
        moe::rhi::ShaderProgram mBrushProgram;
        moe::rhi::ShaderProgram mBuildProgram;
        moe::rhi::ComputePipeline mBrushPipeline;
        moe::rhi::ComputePipeline mBuildPipeline;
        moe::rhi::DescriptorSetLayout mBrushLayout;
        moe::rhi::DescriptorSetLayout mBuildLayout;
        moe::rhi::DescriptorSet mBrushSet;
        moe::rhi::DescriptorSet mBuildSet;
        moe::rhi::CommandList mTerrainCmd;
        moe::neo::Renderer mRenderer;
        moe::neo::SwapchainImage mFrame;
        moe::neo::ProgramHandle mProgram;
        int32_t mPcMvp{-1};

        // CPU mirror of the heights (read back every frame for the camera)
        float mHeights[kVertexCount]{};

        glm::vec3 mPosition{-8.0f, 0.0f, -8.0f};
        float mYaw{0.0f};
        float mPitch{0.0f};
        float mMoveSpeed{4.0f};
        bool mSnowMode{true};
        bool mCaptured{true};
        bool mMoving{false};
        float mBrushRadius{3.2f};
        float mBrushAmount{0.55f};
    };

    // Bilinear terrain height at world XZ (from the CPU mirror).
    float SampleHeight(const SnowData& data, float x, float z) {
        const float u = (x + kTerrainSize * 0.5f) / kCellSize;
        const float v = (z + kTerrainSize * 0.5f) / kCellSize;
        const uint32_t x0 = std::min(static_cast<uint32_t>(u), kGridSize - 1);
        const uint32_t y0 = std::min(static_cast<uint32_t>(v), kGridSize - 1);
        const uint32_t x1 = x0 + 1;
        const uint32_t y1 = y0 + 1;
        const float fx = u - static_cast<float>(x0);
        const float fy = v - static_cast<float>(y0);
        const float h00 = data.mHeights[y0 * kVerticesPerSide + x0];
        const float h10 = data.mHeights[y0 * kVerticesPerSide + x1];
        const float h01 = data.mHeights[y1 * kVerticesPerSide + x0];
        const float h11 = data.mHeights[y1 * kVerticesPerSide + x1];
        return (h00 * (1.0f - fx) + h10 * fx) * (1.0f - fy)
                + (h01 * (1.0f - fx) + h11 * fx) * fy;
    }

    // Deterministic initial terrain (once, on the CPU; afterwards the GPU owns it).
    void InitTerrainHeights(SnowData& data) {
        for (uint32_t y = 0; y <= kGridSize; ++y) {
            for (uint32_t x = 0; x <= kGridSize; ++x) {
                const float fx = static_cast<float>(x) / kGridSize;
                const float fy = static_cast<float>(y) / kGridSize;
                data.mHeights[y * kVerticesPerSide + x] =
                        0.6f * std::sin(fx * 5.0f) * std::cos(fy * 4.0f)
                        + 0.25f * std::sin(fx * 13.0f + 1.7f);
            }
        }
    }

    void BuildTerrainIndices(std::vector<uint32_t>& outIndices) {
        outIndices.reserve(kGridSize * kGridSize * 6);
        for (uint32_t y = 0; y < kGridSize; ++y) {
            for (uint32_t x = 0; x < kGridSize; ++x) {
                const uint32_t i00 = y * kVerticesPerSide + x;
                const uint32_t i10 = y * kVerticesPerSide + x + 1;
                const uint32_t i01 = (y + 1) * kVerticesPerSide + x;
                const uint32_t i11 = (y + 1) * kVerticesPerSide + x + 1;
                // CCW seen from above (Vulkan front face = CCW, y-up)
                outIndices.push_back(i00);
                outIndices.push_back(i01);
                outIndices.push_back(i10);
                outIndices.push_back(i10);
                outIndices.push_back(i01);
                outIndices.push_back(i11);
            }
        }
    }

    // Uploads bytes to a freshly created transfer-dst buffer.
    bool CreateAndUploadBuffer(moe::rhi::Device& device,
            moe::rhi::BufferUsage usage, const void* data, uint64_t byteCount,
            moe::rhi::Buffer& outBuffer) {
        moe::rhi::BufferCreateInfo info{};
        info.mSize = byteCount;
        info.mUsage = usage;
        if (!device.CreateBuffer(info, outBuffer)) {
            return false;
        }

        moe::rhi::BufferCreateInfo stagingInfo{};
        stagingInfo.mSize = byteCount;
        stagingInfo.mUsage = moe::rhi::BufferUsage::kTransferSrc;
        stagingInfo.mCpuVisible = true;
        moe::rhi::Buffer staging;
        if (!device.CreateBuffer(stagingInfo, staging)) {
            return false;
        }
        {
            auto* dst = static_cast<uint8_t*>(staging.Map());
            if (dst == nullptr) {
                moe::Error::Set("staging map failed");
                staging.Destroy();
                return false;
            }
            std::memcpy(dst, data, byteCount);
            staging.Unmap();
        }

        moe::rhi::CommandList cmd;
        if (!device.CreateCommandList(cmd)) {
            staging.Destroy();
            return false;
        }
        cmd.Begin();
        cmd.CopyBuffer(staging, outBuffer, byteCount, 0, 0);
        cmd.End();
        if (!device.Submit(cmd, true)) {
            cmd.Destroy();
            staging.Destroy();
            return false;
        }
        cmd.Destroy();
        staging.Destroy();
        return true;
    }

    bool CreateComputePipeline(moe::rhi::Device& device, moe::rhi::ShaderProgram& program,
            moe::rhi::ComputePipeline& outPipeline) {
        moe::rhi::ComputePipelineState state{};
        state.mProgram = &program;
        if (!device.GetOrCreateComputePipeline(state, outPipeline)) {
            return false;
        }
        return true;
    }

    bool Setup(void* userdata, examples::AppContext& ctx) {
        auto* data = static_cast<SnowData*>(userdata);

        InitTerrainHeights(*data);

        // ---- GPU buffers (heights + indices uploaded once; the vertex buffer
        // is written entirely by the build compute pass) ----
        if (!CreateAndUploadBuffer(ctx.mDevice, moe::rhi::BufferUsage::kStorage
                        | moe::rhi::BufferUsage::kTransferDst | moe::rhi::BufferUsage::kTransferSrc,
                data->mHeights, sizeof(data->mHeights), data->mHeightBuffer)) {
            std::fprintf(stderr, "snow: height buffer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::BufferCreateInfo vertexInfo{};
        vertexInfo.mSize = kVertexCount * kVertexStride;
        vertexInfo.mUsage = moe::rhi::BufferUsage::kStorage | moe::rhi::BufferUsage::kVertex;
        if (!ctx.mDevice.CreateBuffer(vertexInfo, data->mMesh.mVertexBuffer)) {
            std::fprintf(stderr, "snow: vertex buffer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mMesh.mVertexStride = kVertexStride;
        data->mMesh.mVertexCount = kVertexCount;
        data->mMesh.mPositionOffset = kPositionOffset;
        data->mMesh.mNormalOffset = kNormalOffset;
        data->mMesh.mUvOffset = kVertexStride; // absent
        data->mMesh.mColorOffset = kColorOffset;

        std::vector<uint32_t> indices;
        BuildTerrainIndices(indices);
        if (!CreateAndUploadBuffer(ctx.mDevice, moe::rhi::BufferUsage::kIndex
                        | moe::rhi::BufferUsage::kTransferDst,
                indices.data(), indices.size() * sizeof(uint32_t), data->mMesh.mIndexBuffer)) {
            std::fprintf(stderr, "snow: index buffer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mMesh.mIndexCount = static_cast<uint32_t>(indices.size());
        moe::rhi::BufferCreateInfo readbackInfo{};
        readbackInfo.mSize = sizeof(data->mHeights);
        readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
        readbackInfo.mCpuVisible = true;
        if (!ctx.mDevice.CreateBuffer(readbackInfo, data->mHeightReadback)) {
            std::fprintf(stderr, "snow: readback buffer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!ctx.mDevice.CreateCommandList(data->mTerrainCmd)) {
            std::fprintf(stderr, "snow: terrain command list: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // ---- compute pipelines ----
        if (!data->mBrushComp.Load(MOE_SOURCE_DIR "/shaders/examples/snow/snow_brush.comp.spv",
                    moe::rhi::ShaderStage::kCompute)
                || !data->mBuildComp.Load(MOE_SOURCE_DIR "/shaders/examples/snow/snow_build.comp.spv",
                        moe::rhi::ShaderStage::kCompute)
                || !data->mBrushProgram.AddShader(data->mBrushComp)
                || !data->mBuildProgram.AddShader(data->mBuildComp)) {
            std::fprintf(stderr, "snow: compute shader load failed\n");
            return false;
        }
        if (!CreateComputePipeline(ctx.mDevice, data->mBrushProgram, data->mBrushPipeline)
                || !CreateComputePipeline(ctx.mDevice, data->mBuildProgram, data->mBuildPipeline)) {
            std::fprintf(stderr, "snow: compute pipeline: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // ---- descriptor sets: brush reads+writes heights; build reads heights
        // and writes the vertex buffer ----
        if (!data->mBrushPipeline.GetDescriptorSetLayout(0, data->mBrushLayout)
                || !ctx.mDevice.CreateDescriptorSet(data->mBrushLayout, data->mBrushSet)
                || !data->mBrushSet.WriteBuffer(0, data->mHeightBuffer)) {
            std::fprintf(stderr, "snow: brush set: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mBuildPipeline.GetDescriptorSetLayout(0, data->mBuildLayout)
                || !ctx.mDevice.CreateDescriptorSet(data->mBuildLayout, data->mBuildSet)
                || !data->mBuildSet.WriteBuffer(0, data->mHeightBuffer)
                || !data->mBuildSet.WriteBuffer(1, data->mMesh.mVertexBuffer)) {
            std::fprintf(stderr, "snow: build set: %s\n", moe::Error::Get().c_str());
            return false;
        }

        // ---- forward program (content layer) + renderer ----
        data->mProgram = ctx.mAssets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/snow/snow.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/snow/snow.frag.spv");
        if (!data->mProgram.IsValid()) {
            std::fprintf(stderr, "snow: shader load: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!data->mRenderer.Init(ctx.mDevice, ctx.mPipelineCache,
                    ctx.mSwapchain.GetWidth(), ctx.mSwapchain.GetHeight(), ctx.mSampleCount, ctx.mTransfer)) {
            std::fprintf(stderr, "snow: renderer: %s\n", moe::Error::Get().c_str());
            return false;
        }
        data->mPcMvp = data->mRenderer.GetPushConstant(
                *ctx.mAssets.GetProgram(data->mProgram), "mvp");
        if (data->mPcMvp < 0) {
            std::fprintf(stderr, "snow: push constant names mismatch\n");
            return false;
        }

        // ---- input ----
        ctx.mInput.BindAction(kActionMoveForward, static_cast<int32_t>(moe::neo::KeyCode::kW));
        ctx.mInput.BindAction(kActionMoveBack, static_cast<int32_t>(moe::neo::KeyCode::kS));
        ctx.mInput.BindAction(kActionMoveLeft, static_cast<int32_t>(moe::neo::KeyCode::kA));
        ctx.mInput.BindAction(kActionMoveRight, static_cast<int32_t>(moe::neo::KeyCode::kD));
        ctx.mInput.SetMouseCaptured(true);
        return true;
    }

    // Runs the terrain compute chain (brush -> rebuild -> readback) and blocks
    // until the GPU is done, so the camera uses the freshest heights.
    void UpdateTerrain(SnowData& data, moe::rhi::Device& device, float deltaSeconds) {
        moe::rhi::CommandList& cmd = data.mTerrainCmd;

        BrushPushConstants brush{};
        brush.mParams = glm::vec4(
                (data.mPosition.x + kTerrainSize * 0.5f) / kCellSize,
                (data.mPosition.z + kTerrainSize * 0.5f) / kCellSize,
                data.mBrushRadius,
                data.mMoving ? data.mBrushAmount * deltaSeconds : 0.0f);

        cmd.Begin();

        cmd.BindDescriptorSet(data.mBrushPipeline, data.mBrushSet, 0);
        cmd.SetPushConstants(data.mBrushPipeline, 0, sizeof(brush), &brush);
        cmd.Dispatch(data.mBrushPipeline, kVerticesPerSide, kVerticesPerSide, 1);

        // brush write -> build read
        moe::rhi::SyncInfo toBuild{};
        toBuild.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        toBuild.mSrcAccess = moe::rhi::Access::kShaderWrite;
        toBuild.mDstStage = moe::rhi::PipelineStage::kComputeShader;
        toBuild.mDstAccess = moe::rhi::Access::kShaderRead;
        cmd.BufferBarrier(data.mHeightBuffer, toBuild);

        BuildPushConstants build{};
        build.mParams = glm::vec4(data.mSnowMode ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
        cmd.BindDescriptorSet(data.mBuildPipeline, data.mBuildSet, 0);
        cmd.SetPushConstants(data.mBuildPipeline, 0, sizeof(build), &build);
        cmd.Dispatch(data.mBuildPipeline, kVerticesPerSide, kVerticesPerSide, 1);

        // build writes the vertex buffer -> this frame's vertex input reads it
        moe::rhi::SyncInfo toVertexInput{};
        toVertexInput.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        toVertexInput.mSrcAccess = moe::rhi::Access::kShaderWrite;
        toVertexInput.mDstStage = moe::rhi::PipelineStage::kVertexInput;
        toVertexInput.mDstAccess = moe::rhi::Access::kVertexAttributeRead;
        cmd.BufferBarrier(data.mMesh.mVertexBuffer, toVertexInput);

        // heights read (compute) -> readback copy
        moe::rhi::SyncInfo toReadback{};
        toReadback.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        toReadback.mSrcAccess = moe::rhi::Access::kShaderRead;
        toReadback.mDstStage = moe::rhi::PipelineStage::kTransfer;
        toReadback.mDstAccess = moe::rhi::Access::kTransferRead;
        cmd.BufferBarrier(data.mHeightBuffer, toReadback);

        cmd.CopyBuffer(data.mHeightBuffer, data.mHeightReadback, sizeof(data.mHeights), 0, 0);
        cmd.End();

        if (!device.Submit(cmd, true)) {
            std::fprintf(stderr, "snow: terrain submit failed\n");
            return;
        }

        // refresh the CPU mirror for the camera
        const auto* heights = static_cast<const float*>(data.mHeightReadback.Map());
        if (heights != nullptr) {
            std::memcpy(data.mHeights, heights, sizeof(data.mHeights));
            data.mHeightReadback.Unmap();
        }
    }

    void UpdatePlayer(SnowData& data, const moe::neo::Input& input, float deltaSeconds) {
        const moe::neo::MouseState& mouse = input.GetMouse();
        if (mouse.mCaptured) {
            // mouse right = yaw increases = turn right; mouse up = pitch up
            data.mYaw += mouse.mDeltaX * 0.003f;
            data.mPitch = glm::clamp(data.mPitch + mouse.mDeltaY * 0.003f,
                    -1.45f, 1.45f);
        }

        if (mouse.mScrollY != 0.0f) {
            data.mMoveSpeed = glm::clamp(data.mMoveSpeed + mouse.mScrollY * 2.0f,
                    1.0f, 12.0f);
        }

        glm::vec3 dir{0.0f};
        if (input.IsActionDown(kActionMoveForward)) dir.z -= 1.0f;
        if (input.IsActionDown(kActionMoveBack)) dir.z += 1.0f;
        if (input.IsActionDown(kActionMoveLeft)) dir.x -= 1.0f;
        if (input.IsActionDown(kActionMoveRight)) dir.x += 1.0f;
        data.mMoving = dir != glm::vec3(0.0f);
        if (data.mMoving) {
            dir = glm::normalize(dir);
            const float c = std::cos(data.mYaw);
            const float s = std::sin(data.mYaw);
            const glm::vec3 move(dir.x * c + dir.z * s, 0.0f, -dir.x * s + dir.z * c);
            data.mPosition += move * data.mMoveSpeed * deltaSeconds;
            data.mPosition.x = glm::clamp(data.mPosition.x, -kTerrainSize * 0.5f + 0.5f, kTerrainSize * 0.5f - 0.5f);
            data.mPosition.z = glm::clamp(data.mPosition.z, -kTerrainSize * 0.5f + 0.5f, kTerrainSize * 0.5f - 0.5f);
        }
        data.mPosition.y = SampleHeight(data, data.mPosition.x, data.mPosition.z) + 1.7f;
    }

    void DrawIm3d(void* userdata, examples::AppContext& ctx, float deltaSeconds) {
        auto* data = static_cast<SnowData*>(userdata);

        // ---- input + world update (frame start; EndFrame clears deltas later)
        UpdatePlayer(*data, ctx.mInput, deltaSeconds);
        if (ctx.mInput.IsKeyJustPressed(static_cast<int32_t>(moe::neo::KeyCode::kSpace))) {
            data->mCaptured = !data->mCaptured;
            ctx.mInput.SetMouseCaptured(data->mCaptured);
        }
        if (ctx.mInput.IsKeyJustPressed(static_cast<int32_t>(moe::neo::KeyCode::kR))) {
            data->mSnowMode = !data->mSnowMode;
        }

        UpdateTerrain(*data, ctx.mDevice, deltaSeconds);
    }

    void PostRender(void* userdata, examples::AppContext& ctx, moe::rhi::CommandList& cmd) {
        auto* data = static_cast<SnowData*>(userdata);

        const glm::vec3 eye = data->mPosition;
        const glm::vec3 forward(std::cos(data->mPitch) * std::sin(data->mYaw),
                std::sin(data->mPitch),
                -std::cos(data->mPitch) * std::cos(data->mYaw));
        const glm::mat4 view = glm::lookAt(eye, eye + forward, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                static_cast<float>(ctx.mSwapchain.GetWidth())
                        / static_cast<float>(ctx.mSwapchain.GetHeight()),
                0.1f, 200.0f);
        proj[1][1] *= -1; // Vulkan NDC: flip Y
        const glm::mat4 mvp = proj * view;

        const float clear[4] = {0.15f, 0.15f, 0.18f, 1.0f};
        if (!data->mFrame.Acquire(ctx.mSwapchain)) {
            return;
        }
        data->mRenderer.BeginFrame(cmd, data->mFrame, clear);

        const moe::neo::PassDesc pass{"snow", {}, {}};
        data->mRenderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState state;
            state.mCullMode = moe::rhi::CullMode::kNone; // terrain is viewable from below
            context.SetState(state);
            context.SetPushConstant(data->mPcMvp, &mvp, sizeof(mvp));
            context.Draw(data->mMesh, *ctx.mAssets.GetProgram(data->mProgram));
        });

        data->mRenderer.EndFrame();
        data->mFrame.Release();
    }

    void DrawUI(void* userdata, examples::AppContext&) {
        auto* data = static_cast<SnowData*>(userdata);

        ImGui::Begin("snow demo");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("WASD move, mouse look, space toggle cursor");
        ImGui::Text("R snow/sand, scroll wheel: speed %.1f", data->mMoveSpeed);
        if (ImGui::Button(data->mSnowMode ? "snow" : "sand")) {
            data->mSnowMode = !data->mSnowMode;
        }
        ImGui::SliderFloat("brush radius", &data->mBrushRadius, 1.0f, 8.0f);
        ImGui::SliderFloat("brush depth", &data->mBrushAmount, 0.1f, 2.0f);
        ImGui::Checkbox("captured", &data->mCaptured);
        ImGui::End();
    }

    void Shutdown(void* userdata, examples::AppContext&) {
        auto* data = static_cast<SnowData*>(userdata);
        data->mTerrainCmd.Destroy();
        data->mBrushSet.Destroy();
        data->mBuildSet.Destroy();
        data->mHeightReadback.Destroy();
        data->mMesh.Destroy();
        data->mHeightBuffer.Destroy();
        data->mRenderer.Destroy();
    }
}// namespace

int main() {
    SnowData data;
    examples::AppCallbacks callbacks{};
    callbacks.mSetup = Setup;
    callbacks.mDrawIm3d = DrawIm3d;
    callbacks.mPostRender = PostRender;
    callbacks.mDrawUI = DrawUI;
    callbacks.mShutdown = Shutdown;
    callbacks.mUserdata = &data;

    examples::App app;
    if (!app.Run("snow demo", 1280, 720, callbacks)) {
        std::fprintf(stderr, "snow: app: %s\n", moe::Error::Get().c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
