#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

#include "Terrain.hpp"

#include <Core/Error.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace hakoniwa {
    namespace {
        moe::neo::Mesh MakeTerrainMesh(float size, uint32_t resolution, float amplitude) {
            moe::neo::Mesh mesh;
            mesh.mName = "hakoniwa_terrain";
            moe::neo::MeshPrimitive prim;

            const float half = size * 0.5f;
            const float step = size / static_cast<float>(resolution);
            auto height = [&](float x, float z) {
                return amplitude
                        * (0.5f * std::sin(x * 0.012f) * std::cos(z * 0.011f)
                                + 0.3f * std::sin(x * 0.03f + 1.7f)
                                + 0.2f * std::cos(z * 0.025f - 0.6f));
            };

            prim.mPositions.reserve((resolution + 1) * (resolution + 1));
            prim.mNormals.reserve((resolution + 1) * (resolution + 1));
            prim.mUv0.reserve((resolution + 1) * (resolution + 1));
            for (uint32_t z = 0; z <= resolution; ++z) {
                for (uint32_t x = 0; x <= resolution; ++x) {
                    const float wx = -half + static_cast<float>(x) * step;
                    const float wz = -half + static_cast<float>(z) * step;
                    prim.mPositions.push_back(glm::vec3(wx, height(wx, wz), wz));
                    prim.mUv0.push_back(glm::vec2(static_cast<float>(x) / resolution,
                            static_cast<float>(z) / resolution));
                }
            }
            for (uint32_t z = 0; z <= resolution; ++z) {
                for (uint32_t x = 0; x <= resolution; ++x) {
                    const float wx = -half + static_cast<float>(x) * step;
                    const float wz = -half + static_cast<float>(z) * step;
                    const float hl = height(wx - step, wz);
                    const float hr = height(wx + step, wz);
                    const float hd = height(wx, wz - step);
                    const float hu = height(wx, wz + step);
                    prim.mNormals.push_back(
                            glm::normalize(glm::vec3(hl - hr, 2.0f * step, hd - hu)));
                }
            }
            for (uint32_t z = 0; z < resolution; ++z) {
                for (uint32_t x = 0; x < resolution; ++x) {
                    const uint32_t i0 = z * (resolution + 1) + x;
                    const uint32_t i1 = i0 + 1;
                    const uint32_t i2 = i0 + (resolution + 1);
                    const uint32_t i3 = i2 + 1;
                    prim.mIndices.push_back(i0);
                    prim.mIndices.push_back(i2);
                    prim.mIndices.push_back(i1);
                    prim.mIndices.push_back(i1);
                    prim.mIndices.push_back(i2);
                    prim.mIndices.push_back(i3);
                }
            }
            mesh.mPrimitives.push_back(std::move(prim));
            return mesh;
        }
    }// namespace

    bool Terrain::Init(moe::rhi::Device& device, moe::neo::Assets& assets,
            moe::neo::Renderer& renderer, const TerrainParams& params) {
        if (!mUploader.Init(device)) {
            std::fprintf(stderr, "hakoniwa: terrain uploader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!mUploader.UploadMesh(
                    MakeTerrainMesh(params.mSize, params.mResolution, params.mAmplitude), mMesh)) {
            std::fprintf(stderr, "hakoniwa: terrain upload: %s\n", moe::Error::Get().c_str());
            return false;
        }
        mProgram = assets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/terrain.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/terrain.frag.spv");
        if (!mProgram.IsValid()) {
            std::fprintf(stderr, "hakoniwa: terrain shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        const moe::rhi::ShaderProgram* program = assets.GetProgram(mProgram);
        mPcViewProj = renderer.GetPushConstant(*program, "viewProj");
        mPcModel = renderer.GetPushConstant(*program, "model");
        mPcCameraPos = renderer.GetPushConstant(*program, "cameraPos");
        mPcSunDir = renderer.GetPushConstant(*program, "sunDir");
        mPcSunColor = renderer.GetPushConstant(*program, "sunColor");
        mPcAmbient = renderer.GetPushConstant(*program, "ambient");
        mPcFogColor = renderer.GetPushConstant(*program, "fogColor");
        mPcFogDensity = renderer.GetPushConstant(*program, "fogDensity");
        if (mPcViewProj < 0 || mPcModel < 0 || mPcCameraPos < 0 || mPcSunDir < 0
                || mPcSunColor < 0 || mPcAmbient < 0 || mPcFogColor < 0 || mPcFogDensity < 0) {
            std::fprintf(stderr, "hakoniwa: terrain push constant names mismatch\n");
            return false;
        }
        return true;
    }

    void Terrain::Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
            moe::neo::RenderTargetHandle target, const TerrainFrame& frame,
            const TerrainParams& params) {
        const glm::mat4 model(1.0f);
        const moe::neo::PassDesc pass{"hakoniwa terrain",
                moe::neo::ColorAttachment(target, moe::rhi::LoadOp::kLoad),
                moe::neo::DepthAttachment(target, moe::rhi::LoadOp::kLoad)};
        renderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState state;
            state.mDepthTest = true;
            state.mDepthWrite = true;
            state.mCullMode = moe::rhi::CullMode::kNone;
            context.SetState(state);
            context.ClearTextureBindings();
            context.SetPushConstant(mPcViewProj, &frame.mViewProj, sizeof(glm::mat4));
            context.SetPushConstant(mPcModel, &model, sizeof(glm::mat4));
            context.SetPushConstant(mPcCameraPos, &frame.mCameraPos, sizeof(glm::vec3));
            context.SetPushConstant(mPcSunDir, &frame.mSunDir, sizeof(glm::vec3));
            context.SetPushConstant(mPcSunColor, &frame.mSunColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcAmbient, &frame.mAmbient, sizeof(glm::vec3));
            context.SetPushConstant(mPcFogColor, &frame.mFogColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcFogDensity, &params.mFogDensity, sizeof(float));
            context.Draw(mMesh, *assets.GetProgram(mProgram));
        });
    }

    void Terrain::RecordMinimap(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
            moe::neo::RenderTargetHandle target, const TerrainParams& params) {
        const float half = params.mSize * 0.5f;
        constexpr float kCameraY = 900.0f;
        constexpr float kNear = 0.1f;
        constexpr float kFar = 2000.0f;
        const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, kCameraY, 0.0f), glm::vec3(0.0f),
                glm::vec3(0.0f, 0.0f, -1.0f));
        const glm::mat4 viewProj = glm::ortho(-half, half, -half, half, kNear, kFar) * view;

        const glm::mat4 model(1.0f);
        const glm::vec3 cameraPos(0.0f, kCameraY, 0.0f);
        const glm::vec3 sunDir(0.0f, 1.0f, 0.0f);
        const glm::vec3 sunColor(1.0f);
        const glm::vec3 ambient(0.5f);
        const glm::vec3 fogColor(0.0f);
        const float noFog = 0.0f;
        const moe::neo::PassDesc pass{"hakoniwa minimap terrain",
                moe::neo::ColorAttachment(target, moe::rhi::LoadOp::kClear),
                moe::neo::DepthAttachment(target, moe::rhi::LoadOp::kClear)};
        renderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState state;
            state.mDepthTest = true;
            state.mDepthWrite = true;
            state.mCullMode = moe::rhi::CullMode::kNone;
            context.SetState(state);
            context.ClearTextureBindings();
            context.SetPushConstant(mPcViewProj, &viewProj, sizeof(glm::mat4));
            context.SetPushConstant(mPcModel, &model, sizeof(glm::mat4));
            context.SetPushConstant(mPcCameraPos, &cameraPos, sizeof(glm::vec3));
            context.SetPushConstant(mPcSunDir, &sunDir, sizeof(glm::vec3));
            context.SetPushConstant(mPcSunColor, &sunColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcAmbient, &ambient, sizeof(glm::vec3));
            context.SetPushConstant(mPcFogColor, &fogColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcFogDensity, &noFog, sizeof(float));
            context.Draw(mMesh, *assets.GetProgram(mProgram));
        });
    }

    void Terrain::Destroy() {
        mMesh.Destroy();
    }
}// namespace hakoniwa
