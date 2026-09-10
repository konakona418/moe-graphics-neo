#include "UI/Im3dDrawer.hpp"

#include <Core/Error.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Device.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/Shader.hpp>
#include <RHI/Swapchain.hpp>

#include <im3d.h>

#include <cstdio>
#include <cstring>

namespace moe::ui {
    namespace {
        constexpr uint32_t kMaxVertexCount = 65536;

        // Mirrors shaders/slang/im3d/im3d_common.slang's Im3dPCS (80 bytes
        // with scalar layout): vertex buffer device address + view-proj +
        // viewport size.
        struct PushConstants {
            uint64_t mVertexBufferAddress{0};
            glm::mat4 mViewProj{1.0f};
            glm::vec2 mViewport{0.0f, 0.0f};
        };

        struct PipelineSet {
            moe::rhi::ShaderProgram mProgram;
            moe::rhi::GraphicsPipeline mPipeline;
        };
    }// namespace

    struct Im3dDrawerImpl {
        moe::rhi::Buffer mVertexBuffer;
        moe::rhi::Buffer mStaging;
        PipelineSet mPoint;
        PipelineSet mLine;
        PipelineSet mTriangle;
        bool mActive{false};
    };

    Im3dDrawer::Im3dDrawer() = default;

    Im3dDrawer::~Im3dDrawer() {
        if (mImpl != nullptr) {
            std::fprintf(stderr, "[ui] Im3dDrawer leaked: Destroy() not called\n");
            std::abort();
        }
    }

    void Im3dDrawer::Destroy() {
        if (mImpl == nullptr) {
            return;
        }
        mImpl->mVertexBuffer.Destroy();
        mImpl->mStaging.Destroy();
        mImpl.reset();
    }

    bool Im3dDrawer::IsActive() const {
        return mImpl != nullptr && mImpl->mActive;
    }

    bool Im3dDrawer::HasDraws() const {
        if (!IsActive()) {
            return false;
        }
        const Im3d::DrawList* lists = Im3d::GetDrawLists();
        for (uint32_t i = 0; i < Im3d::GetDrawListCount(); ++i) {
            if (lists[i].m_vertexCount > 0) {
                return true;
            }
        }
        return false;
    }

    bool Im3dDrawer::Init(moe::rhi::Device& device, moe::rhi::DefaultPipelineCache& cache,
            moe::rhi::Swapchain& swapchain) {
        mImpl = std::make_unique<Im3dDrawerImpl>();

        struct ShaderPaths {
            const char* mVert;
            const char* mGeom;
            const char* mFrag;
        };
        const ShaderPaths paths[] = {
                {MOE_SOURCE_DIR "/shaders/im3d/point.vert.spv", nullptr, MOE_SOURCE_DIR "/shaders/im3d/point.frag.spv"},
                {MOE_SOURCE_DIR "/shaders/im3d/line.vert.spv", MOE_SOURCE_DIR "/shaders/im3d/line.geom.spv", MOE_SOURCE_DIR "/shaders/im3d/line.frag.spv"},
                {MOE_SOURCE_DIR "/shaders/im3d/triangle.vert.spv", nullptr, MOE_SOURCE_DIR "/shaders/im3d/triangle.frag.spv"},
        };
        const moe::rhi::PrimitiveTopology topologies[] = {
                moe::rhi::PrimitiveTopology::kPointList,
                moe::rhi::PrimitiveTopology::kLineList,
                moe::rhi::PrimitiveTopology::kTriangleList,
        };
        PipelineSet* outPipelines[] = {&mImpl->mPoint, &mImpl->mLine, &mImpl->mTriangle};

        for (uint32_t i = 0; i < 3; ++i) {
            moe::rhi::Shader vert;
            moe::rhi::Shader geom;
            moe::rhi::Shader frag;
            if (!vert.Load(paths[i].mVert, moe::rhi::ShaderStage::kVertex)
                    || !frag.Load(paths[i].mFrag, moe::rhi::ShaderStage::kFragment)) {
                mImpl.reset();
                return moe::Fail("Im3d shader load failed: " + moe::Error::Get());
            }
            if (paths[i].mGeom != nullptr
                    && !geom.Load(paths[i].mGeom, moe::rhi::ShaderStage::kGeometry)) {
                mImpl.reset();
                return moe::Fail("Im3d geometry shader load failed: " + moe::Error::Get());
            }
            if (!outPipelines[i]->mProgram.AddShader(vert)
                    || (paths[i].mGeom != nullptr && !outPipelines[i]->mProgram.AddShader(geom))
                    || !outPipelines[i]->mProgram.AddShader(frag)) {
                mImpl.reset();
                return moe::Fail("Im3d program add failed");
            }

            moe::rhi::GraphicsPipelineState state{};
            state.mProgram = &outPipelines[i]->mProgram;
            state.mTopology = topologies[i];
            state.mRaster.mCullMode = moe::rhi::CullMode::kNone;
            // alpha blend (same as the old engine's enableBlending defaults)
            state.mBlendAttachmentCount = 1;
            state.mBlendAttachments[0].mBlendEnabled = true;
            state.mBlendAttachments[0].mSrcColor = moe::rhi::BlendFactor::kSrcAlpha;
            state.mBlendAttachments[0].mDstColor = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            state.mBlendAttachments[0].mSrcAlpha = moe::rhi::BlendFactor::kOne;
            state.mBlendAttachments[0].mDstAlpha = moe::rhi::BlendFactor::kOneMinusSrcAlpha;
            // depth test LESS + write (mirrors the old engine; Im3d draws last
            // in the pass so the write has no downstream effect)
            state.mDepth.mTestEnable = true;
            state.mDepth.mWriteEnable = true;
            state.mDepth.mCompareOp = moe::rhi::CompareOp::kLess;
            state.mDepthFormat = moe::rhi::Format::kD32Float;
            state.mColorFormatCount = 1;
            state.mColorFormats[0] = swapchain.GetFormat();
            if (!device.GetOrCreateGraphicsPipeline(state, outPipelines[i]->mPipeline)) {
                mImpl.reset();
                return moe::Fail("Im3d pipeline: " + moe::Error::Get());
            }
        }

        moe::rhi::BufferCreateInfo bufferInfo{};
        bufferInfo.mSize = sizeof(Im3d::VertexData) * kMaxVertexCount;
        bufferInfo.mUsage = moe::rhi::BufferUsage::kVertex | moe::rhi::BufferUsage::kTransferDst;
        if (!device.CreateBuffer(bufferInfo, mImpl->mVertexBuffer)) {
            mImpl.reset();
            return moe::Fail("Im3d vertex buffer: " + moe::Error::Get());
        }
        bufferInfo.mUsage = moe::rhi::BufferUsage::kTransferSrc;
        bufferInfo.mCpuVisible = true;
        if (!device.CreateBuffer(bufferInfo, mImpl->mStaging)) {
            mImpl.reset();
            return moe::Fail("Im3d staging buffer: " + moe::Error::Get());
        }

        mImpl->mActive = true;
        return true;
    }

    void Im3dDrawer::UploadVertices(moe::rhi::CommandList& cmd) {
        if (!HasDraws()) {
            return;
        }

        // Gather the per-draw-list vertices into the staging buffer.
        const Im3d::DrawList* drawLists = Im3d::GetDrawLists();
        size_t vertexCountAccum = 0;
        auto* staging = static_cast<uint8_t*>(mImpl->mStaging.Map());
        if (staging == nullptr) {
            return;
        }
        for (uint32_t i = 0; i < Im3d::GetDrawListCount(); ++i) {
            const Im3d::DrawList& dl = drawLists[i];
            if (vertexCountAccum + dl.m_vertexCount > kMaxVertexCount) {
                std::fprintf(stderr, "[ui] Im3d vertex count exceeds %u, clamping\n",
                        static_cast<unsigned>(kMaxVertexCount));
                break;
            }
            std::memcpy(staging + vertexCountAccum * sizeof(Im3d::VertexData),
                    dl.m_vertexData, dl.m_vertexCount * sizeof(Im3d::VertexData));
            vertexCountAccum += dl.m_vertexCount;
        }
        mImpl->mStaging.Unmap();

        const uint64_t bytes = vertexCountAccum * sizeof(Im3d::VertexData);
        if (bytes == 0) {
            return;
        }
        cmd.CopyBuffer(mImpl->mStaging, mImpl->mVertexBuffer, bytes, 0, 0);

        // staging (transfer) write -> vertex shader read (device address)
        moe::rhi::SyncInfo sync{};
        sync.mSrcStage = moe::rhi::PipelineStage::kTransfer;
        sync.mSrcAccess = moe::rhi::Access::kTransferWrite;
        sync.mDstStage = moe::rhi::PipelineStage::kVertexShader;
        sync.mDstAccess = moe::rhi::Access::kShaderRead;
        cmd.BufferBarrier(mImpl->mVertexBuffer, sync);
    }

    void Im3dDrawer::Record(moe::rhi::CommandList& cmd) {
        if (!HasDraws()) {
            return;
        }

        const Im3d::DrawList* drawLists = Im3d::GetDrawLists();
        PushConstants pcs{};
        pcs.mVertexBufferAddress = mImpl->mVertexBuffer.GetDeviceAddress();
        pcs.mViewProj = mViewProj;
        pcs.mViewport = mViewport;
        cmd.SetViewport(static_cast<uint32_t>(mViewport.x), static_cast<uint32_t>(mViewport.y));

        const PipelineSet* currentSet = nullptr;
        uint32_t vertexOffset = 0;
        for (uint32_t i = 0; i < Im3d::GetDrawListCount(); ++i) {
            const Im3d::DrawList& dl = drawLists[i];
            if (dl.m_vertexCount == 0) {
                continue;
            }
            const PipelineSet* set = nullptr;
            switch (dl.m_primType) {
                case Im3d::DrawPrimitive_Points: set = &mImpl->mPoint; break;
                case Im3d::DrawPrimitive_Lines: set = &mImpl->mLine; break;
                case Im3d::DrawPrimitive_Triangles: set = &mImpl->mTriangle; break;
                default: continue;
            }
            if (set != currentSet) {
                cmd.BindGraphicsPipeline(set->mPipeline);
                cmd.SetPushConstants(set->mPipeline, 0, sizeof(PushConstants), &pcs);
                currentSet = set;
            }
            cmd.Draw(dl.m_vertexCount, 1, vertexOffset, 0);
            vertexOffset += dl.m_vertexCount;
        }
    }
}// namespace moe::ui
