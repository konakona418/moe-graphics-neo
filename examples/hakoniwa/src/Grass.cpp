#include "Grass.hpp"

#include <Core/Error.hpp>

#include <cstdio>
#include <vector>

namespace hakoniwa {
    namespace {
        constexpr uint32_t kFieldResolution = 128;

        // Must match grass_displacement.slang's PushConstants layout.
        struct DisplacePushConstants {
            float mOriginX;
            float mOriginY;
            float mSize;
            float mRadius;
            float mStrength;
            float mDecay;
            float mPad0;
            float mPad1;
        };

        // A tapered blade in local space: x in [-0.5, 0.5] is the width, y in
        // [0, 1] is the height fraction (the vertex shader scales and bends it).
        moe::neo::Mesh MakeBladeMesh() {
            moe::neo::Mesh mesh;
            mesh.mName = "hakoniwa_grass_blade";
            moe::neo::MeshPrimitive prim;

            constexpr int kRows = 4;
            prim.mPositions.reserve(kRows * 2);
            for (int row = 0; row < kRows; ++row) {
                const float t = static_cast<float>(row) / static_cast<float>(kRows - 1);
                const float halfWidth = 0.5f * (1.0f - 0.9f * t);
                prim.mPositions.push_back(glm::vec3(-halfWidth, t, 0.0f));
                prim.mPositions.push_back(glm::vec3(halfWidth, t, 0.0f));
            }
            for (int row = 0; row < kRows - 1; ++row) {
                const uint32_t base = static_cast<uint32_t>(row) * 2;
                prim.mIndices.push_back(base);
                prim.mIndices.push_back(base + 2);
                prim.mIndices.push_back(base + 1);
                prim.mIndices.push_back(base + 1);
                prim.mIndices.push_back(base + 2);
                prim.mIndices.push_back(base + 3);
            }
            mesh.mPrimitives.push_back(std::move(prim));
            return mesh;
        }
    }// namespace

    bool Grass::Init(moe::rhi::Device& device, moe::neo::Assets& assets,
            moe::neo::Renderer& renderer) {
        if (!mUploader.Init(device)) {
            std::fprintf(stderr, "hakoniwa: grass uploader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!mUploader.UploadMesh(MakeBladeMesh(), mMesh)) {
            std::fprintf(stderr, "hakoniwa: grass mesh: %s\n", moe::Error::Get().c_str());
            return false;
        }
        mProgram = assets.LoadGraphicsProgram(
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/grass.vert.spv",
                MOE_SOURCE_DIR "/shaders/examples/hakoniwa/grass.frag.spv");
        if (!mProgram.IsValid()) {
            std::fprintf(stderr, "hakoniwa: grass shader: %s\n", moe::Error::Get().c_str());
            return false;
        }
        const moe::rhi::ShaderProgram* program = assets.GetProgram(mProgram);
        mPcViewProj = renderer.GetPushConstant(*program, "viewProj");
        mPcCameraPos = renderer.GetPushConstant(*program, "cameraPos");
        mPcSunDir = renderer.GetPushConstant(*program, "sunDir");
        mPcSunColor = renderer.GetPushConstant(*program, "sunColor");
        mPcAmbient = renderer.GetPushConstant(*program, "ambient");
        mPcFogColor = renderer.GetPushConstant(*program, "fogColor");
        mPcFogDensity = renderer.GetPushConstant(*program, "fogDensity");
        mPcAmplitude = renderer.GetPushConstant(*program, "amplitude");
        mPcTileSize = renderer.GetPushConstant(*program, "tileSize");
        mPcTilesPerSide = renderer.GetPushConstant(*program, "tilesPerSide");
        mPcBladesPerTile = renderer.GetPushConstant(*program, "bladesPerTile");
        mPcInnerTiles = renderer.GetPushConstant(*program, "innerTiles");
        mPcOuterTiles = renderer.GetPushConstant(*program, "outerTiles");
        mPcWindStrength = renderer.GetPushConstant(*program, "windStrength");
        mPcTime = renderer.GetPushConstant(*program, "time");
        mPcTerrainHalfSize = renderer.GetPushConstant(*program, "terrainHalfSize");
        mPcFieldOrigin = renderer.GetPushConstant(*program, "fieldOrigin");
        mPcFieldSize = renderer.GetPushConstant(*program, "fieldSize");
        mPcFieldDisplace = renderer.GetPushConstant(*program, "fieldDisplace");
        if (mPcViewProj < 0 || mPcCameraPos < 0 || mPcSunDir < 0
                || mPcSunColor < 0 || mPcAmbient < 0 || mPcFogColor < 0 || mPcFogDensity < 0
                || mPcAmplitude < 0 || mPcTileSize < 0 || mPcTilesPerSide < 0
                || mPcBladesPerTile < 0 || mPcInnerTiles < 0 || mPcOuterTiles < 0
                || mPcWindStrength < 0 || mPcTime < 0 || mPcTerrainHalfSize < 0
                || mPcFieldOrigin < 0 || mPcFieldSize < 0 || mPcFieldDisplace < 0) {
            std::fprintf(stderr, "hakoniwa: grass push constant names mismatch\n");
            return false;
        }

        // ---- player displacement field ----
        moe::rhi::ImageCreateInfo fieldInfo{};
        fieldInfo.mType = moe::rhi::ImageType::k2D;
        fieldInfo.mWidth = kFieldResolution;
        fieldInfo.mHeight = kFieldResolution;
        fieldInfo.mFormat = moe::rhi::Format::kR16G16B16A16Float;
        fieldInfo.mUsage = moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled;
        if (!device.CreateImage(fieldInfo, mField)) {
            std::fprintf(stderr, "hakoniwa: grass field image: %s\n", moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::SamplerCreateInfo fieldSampler{};
        fieldSampler.mAddressModeU = moe::rhi::AddressMode::kClampToEdge;
        fieldSampler.mAddressModeV = moe::rhi::AddressMode::kClampToEdge;
        fieldSampler.mAddressModeW = moe::rhi::AddressMode::kClampToEdge;
        if (!device.CreateSampler(fieldSampler, mFieldSampler)) {
            std::fprintf(stderr, "hakoniwa: grass field sampler: %s\n", moe::Error::Get().c_str());
            return false;
        }
        if (!mDisplaceComp.Load(
                    MOE_SOURCE_DIR "/shaders/examples/hakoniwa/grass_displacement.comp.spv",
                    moe::rhi::ShaderStage::kCompute)
                || !mDisplaceProgram.AddShader(mDisplaceComp)) {
            std::fprintf(stderr, "hakoniwa: grass displacement shader: %s\n",
                    moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::ComputePipelineState displaceState{};
        displaceState.mProgram = &mDisplaceProgram;
        if (!device.GetOrCreateComputePipeline(displaceState, mDisplacePipeline)) {
            std::fprintf(stderr, "hakoniwa: grass displacement pipeline: %s\n",
                    moe::Error::Get().c_str());
            return false;
        }
        moe::rhi::DescriptorSetLayout displaceLayout;
        if (!mDisplacePipeline.GetDescriptorSetLayout(0, displaceLayout)
                || !device.CreateDescriptorSet(displaceLayout, mDisplaceSet)
                || !mDisplaceSet.WriteImage(0, mField, moe::rhi::DescriptorType::kStorageImage)) {
            std::fprintf(stderr, "hakoniwa: grass displacement set: %s\n",
                    moe::Error::Get().c_str());
            return false;
        }
        return true;
    }

    void Grass::Record(moe::rhi::CommandList& cmd, moe::neo::Assets& assets,
            moe::neo::Renderer& renderer, moe::neo::RenderTargetHandle target,
            const GrassFrame& frame, const GrassParams& params) {
        if (params.mTilesPerSide <= 0 || params.mBladesPerTile <= 0) {
            return;
        }

        // ---- 1. displacement field: decay + radial splat around the player ----
        moe::rhi::SyncInfo toCompute{};
        toCompute.mSrcStage = mFieldLayout == moe::rhi::ImageLayout::kUndefined
                ? moe::rhi::PipelineStage::kTopOfPipe
                : moe::rhi::PipelineStage::kVertexShader;
        toCompute.mSrcAccess = mFieldLayout == moe::rhi::ImageLayout::kUndefined
                ? moe::rhi::Access::kNone
                : moe::rhi::Access::kShaderRead;
        toCompute.mDstStage = moe::rhi::PipelineStage::kComputeShader;
        toCompute.mDstAccess = moe::rhi::Access::kShaderWrite;
        renderer.ImageBarrier(mField, mFieldLayout, moe::rhi::ImageLayout::kGeneral, toCompute);
        mFieldLayout = moe::rhi::ImageLayout::kGeneral;

        DisplacePushConstants displace{};
        displace.mOriginX = frame.mPlayerPos.x;
        displace.mOriginY = frame.mPlayerPos.z;
        displace.mSize = params.mFieldSize;
        displace.mRadius = params.mFieldRadius;
        displace.mStrength = params.mFieldStrength;
        displace.mDecay = params.mFieldDecay;
        cmd.BindDescriptorSet(mDisplacePipeline, mDisplaceSet, 0);
        cmd.SetPushConstants(mDisplacePipeline, 0, sizeof(displace), &displace);
        cmd.Dispatch(mDisplacePipeline, (kFieldResolution + 7) / 8, (kFieldResolution + 7) / 8, 1);

        moe::rhi::SyncInfo toGrass{};
        toGrass.mSrcStage = moe::rhi::PipelineStage::kComputeShader;
        toGrass.mSrcAccess = moe::rhi::Access::kShaderWrite;
        toGrass.mDstStage = moe::rhi::PipelineStage::kVertexShader;
        toGrass.mDstAccess = moe::rhi::Access::kShaderRead;
        renderer.ImageBarrier(mField, moe::rhi::ImageLayout::kGeneral,
                moe::rhi::ImageLayout::kShaderReadOnly, toGrass);
        mFieldLayout = moe::rhi::ImageLayout::kShaderReadOnly;

        // ---- 2. the blade field, one instanced draw per clipmap ring ----
        const moe::neo::PassDesc pass{"hakoniwa grass",
                moe::neo::ColorAttachment(target, moe::rhi::LoadOp::kLoad),
                moe::neo::DepthAttachment(target, moe::rhi::LoadOp::kLoad)};
        renderer.Execute(pass, [&](moe::neo::PassContext& context) {
            moe::neo::DrawState state;
            state.mDepthTest = true;
            state.mDepthWrite = true;
            state.mCullMode = moe::rhi::CullMode::kNone;
            context.SetState(state);
            context.ClearTextureBindings();
            context.BindImage(0, mField);
            context.BindSampler(1, mFieldSampler);
            context.SetPushConstant(mPcViewProj, &frame.mViewProj, sizeof(glm::mat4));
            context.SetPushConstant(mPcCameraPos, &frame.mCameraPos, sizeof(glm::vec3));
            context.SetPushConstant(mPcSunDir, &frame.mSunDir, sizeof(glm::vec3));
            context.SetPushConstant(mPcSunColor, &frame.mSunColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcAmbient, &frame.mAmbient, sizeof(glm::vec3));
            context.SetPushConstant(mPcFogColor, &frame.mFogColor, sizeof(glm::vec3));
            context.SetPushConstant(mPcFogDensity, &frame.mFogDensity, sizeof(float));
            context.SetPushConstant(mPcAmplitude, &frame.mTerrainAmplitude, sizeof(float));
            context.SetPushConstant(mPcWindStrength, &params.mWindStrength, sizeof(float));
            context.SetPushConstant(mPcTime, &frame.mTime, sizeof(float));
            context.SetPushConstant(mPcTerrainHalfSize, &frame.mTerrainHalfSize, sizeof(float));
            const glm::vec2 fieldOrigin(frame.mPlayerPos.x, frame.mPlayerPos.z);
            context.SetPushConstant(mPcFieldOrigin, &fieldOrigin, sizeof(glm::vec2));
            context.SetPushConstant(mPcFieldSize, &params.mFieldSize, sizeof(float));
            context.SetPushConstant(mPcFieldDisplace, &params.mFieldDisplace, sizeof(float));

            const moe::rhi::ShaderProgram& program = *assets.GetProgram(mProgram);
            const int32_t ringCount = params.mRingCount < 1 ? 1 : params.mRingCount;
            const int32_t side = params.mTilesPerSide;
            const int32_t blades = params.mBladesPerTile;
            const uint32_t instanceCount = static_cast<uint32_t>(side)
                    * static_cast<uint32_t>(side) * static_cast<uint32_t>(blades);
            for (int32_t ring = 0; ring < ringCount; ++ring) {
                const float tileSize = params.mBaseTileSize
                        * static_cast<float>(1u << static_cast<uint32_t>(ring));
                const int32_t innerTiles = ring == 0 ? 0 : side / 4;
                const int32_t outerTiles = side / 2;
                context.SetPushConstant(mPcTileSize, &tileSize, sizeof(float));
                context.SetPushConstant(mPcTilesPerSide, &side, sizeof(int32_t));
                context.SetPushConstant(mPcBladesPerTile, &blades, sizeof(int32_t));
                context.SetPushConstant(mPcInnerTiles, &innerTiles, sizeof(int32_t));
                context.SetPushConstant(mPcOuterTiles, &outerTiles, sizeof(int32_t));
                context.Draw(mMesh, program, moe::rhi::PrimitiveTopology::kTriangleList,
                        instanceCount);
            }
        });
    }

    void Grass::Destroy() {
        mDisplaceSet.Destroy();
        mFieldSampler.Destroy();
        mField.Destroy();
        mMesh.Destroy();
    }
}// namespace hakoniwa
