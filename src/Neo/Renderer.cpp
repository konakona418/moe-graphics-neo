#include "Neo/Renderer.hpp"

#include <Core/Logger.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Image.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <vector>

namespace moe::neo {
    namespace {
        constexpr uint32_t kMaxPushConstantBytes = 256;
        constexpr uint32_t kMaxDrawsPerFrame = 1024;
        constexpr uint32_t kMaxInstanceAttributes = 16;

        uint32_t FormatSize(rhi::Format format) {
            switch (format) {
                case rhi::Format::kR32Float:
                case rhi::Format::kR32Uint:
                case rhi::Format::kR8G8B8A8Unorm:
                case rhi::Format::kR8G8B8A8Srgb: return 4;
                case rhi::Format::kR32G32Float: return 8;
                case rhi::Format::kR32G32B32Float: return 12;
                case rhi::Format::kR32G32B32A32Float:
                case rhi::Format::kR16G16B16A16Float:
                case rhi::Format::kR16G16Float: return 16;
                default: return 16;
            }
        }
    }// namespace

    struct Renderer::Impl {
        rhi::Device* mDevice{nullptr};
        rhi::DefaultPipelineCache* mCache{nullptr};
        rhi::Image mDepthImage;
        uint32_t mWidth{0};
        uint32_t mHeight{0};
        std::string mLastError;

        // push constant name registry (lookup-once, then set by index)
        struct FieldReg {
            const rhi::ShaderProgram* mProgram{nullptr};
            std::string mName;
            uint32_t mOffset{0};
            uint32_t mSize{0};
        };
        mutable std::vector<FieldReg> mFields;

        // ---- per-frame GL-style state (reset in BeginFrame; each Draw
        // snapshots what it sees) ----

        struct ImageBinding {
            uint32_t mBinding{0};
            const rhi::Image* mImage{nullptr};
        };
        struct SamplerBinding {
            uint32_t mBinding{0};
            const rhi::Sampler* mSampler{nullptr};
        };
        struct InstanceBind {
            const rhi::Buffer* mBuffer{nullptr};
            uint32_t mStride{0};
            std::array<InstanceAttribute, kMaxInstanceAttributes> mAttributes{};
            uint32_t mAttributeCount{0};
        };

        DrawState mState;
        RenderTargetHandle mTarget;
        RenderTarget* mTargetPtr{nullptr};
        std::array<ImageBinding, kMaxTextureBindings> mImages{};
        uint32_t mImageCount{0};
        std::array<SamplerBinding, kMaxTextureBindings> mSamplers{};
        uint32_t mSamplerCount{0};
        InstanceBind mInstance;
        std::array<uint8_t, kMaxPushConstantBytes> mTable{};
        uint32_t mTableSize{0};

        // frame state
        rhi::CommandList* mCmd{nullptr};
        // Borrowed swapchain image: owned by the caller, valid from
        // BeginFrame to EndFrame (both inside the same App callback scope).
        const rhi::Image* mSwapchainImage{nullptr};
        rhi::Format mSwapchainFormat{rhi::Format::kUndefined};
        float mClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        bool mDepthReady{false};

        Cache<RenderTarget> mTargets;

        // Descriptor sets created during EndFrame recording; freed at the
        // next BeginFrame (queue is serial, so in-flight usage has finished).
        std::vector<std::unique_ptr<rhi::DescriptorSet>> mFrameSets;

        // per-second frame statistics
        std::chrono::steady_clock::time_point mStatsTime{};
        uint32_t mStatsFrames{0};
        uint64_t mStatsDraws{0};
        uint32_t mStatsPasses{0};
        uint32_t mStatsLastDraws{0};

        struct DrawCmd {
            const rhi::ShaderProgram* mProgram{nullptr};
            const UploadedMesh* mMesh{nullptr}; // null = fullscreen
            rhi::PrimitiveTopology mTopology{rhi::PrimitiveTopology::kTriangleList};
            uint32_t mInstanceCount{1};
            DrawState mState;
            RenderTargetHandle mTarget;
            RenderTarget* mTargetPtr{nullptr};
            std::array<ImageBinding, kMaxTextureBindings> mImages{};
            uint32_t mImageCount{0};
            std::array<SamplerBinding, kMaxTextureBindings> mSamplers{};
            uint32_t mSamplerCount{0};
            InstanceBind mInstance;
            std::array<uint8_t, kMaxPushConstantBytes> mTable{};
            uint32_t mTableSize{0};
            uint64_t mPipelineHash{0};
        };
        std::vector<DrawCmd> mDraws;

        RenderTarget* ResolveTarget(RenderTargetHandle handle) {
            return handle.IsValid() ? mTargets.Get(handle) : nullptr;
        }

        // Resolves a push constant member of the program by name (registry
        // lookup or reflection; registers on first hit).
        int32_t LookupField(const rhi::ShaderProgram& program, const char* name) {
            for (size_t i = 0; i < mFields.size(); ++i) {
                const auto& field = mFields[i];
                if (field.mProgram == &program && field.mName == name) {
                    return static_cast<int32_t>(i);
                }
            }
            const rhi::ShaderStage stages[3] = {
                    rhi::ShaderStage::kVertex, rhi::ShaderStage::kFragment, rhi::ShaderStage::kGeometry};
            for (const rhi::ShaderStage stage : stages) {
                const rhi::Shader* shader = program.GetStage(stage);
                if (shader == nullptr) {
                    continue;
                }
                for (const auto& field : shader->GetReflection().mPushConstantFields) {
                    if (field.mName == name) {
                        mFields.push_back({&program, name, field.mOffset, field.mSize});
                        return static_cast<int32_t>(mFields.size()) - 1;
                    }
                }
            }
            return -1;
        }

        // Builds the RHI pipeline state for one queued draw: vertex layout
        // from the uploaded mesh (binding 0) + explicit instance attributes
        // (binding 1), state from the draw's snapshot, formats from the
        // render target.
        rhi::GraphicsPipelineState BuildPipelineState(const DrawCmd& cmd) {
            rhi::GraphicsPipelineState state{};
            state.mProgram = cmd.mProgram;
            state.mTopology = cmd.mTopology;

            state.mColorFormatCount = 1;
            state.mColorFormats[0] =
                    cmd.mTargetPtr != nullptr ? cmd.mTargetPtr->mFormat : mSwapchainFormat;
            state.mDepthFormat = rhi::Format::kD32Float;

            state.mBlendAttachmentCount = 1;
            state.mBlendAttachments[0].mBlendEnabled = cmd.mState.mBlendEnabled;
            state.mBlendAttachments[0].mSrcColor = cmd.mState.mBlendSrcColor;
            state.mBlendAttachments[0].mDstColor = cmd.mState.mBlendDstColor;
            state.mBlendAttachments[0].mColorOp = cmd.mState.mBlendColorOp;
            state.mBlendAttachments[0].mSrcAlpha = cmd.mState.mBlendSrcAlpha;
            state.mBlendAttachments[0].mDstAlpha = cmd.mState.mBlendDstAlpha;
            state.mBlendAttachments[0].mAlphaOp = cmd.mState.mBlendAlphaOp;

            state.mRaster.mCullMode = cmd.mState.mCullMode;
            state.mRaster.mFrontFace = cmd.mState.mFrontFace;
            state.mRaster.mPolygonMode = cmd.mState.mPolygonMode;

            state.mDepth.mTestEnable = cmd.mState.mDepthTest;
            state.mDepth.mWriteEnable = cmd.mState.mDepthWrite;
            state.mDepth.mCompareOp = cmd.mState.mDepthCompareOp;

            uint32_t attributeCount = 0;
            if (cmd.mMesh != nullptr) {
                const UploadedMesh& mesh = *cmd.mMesh;
                state.mVertexBindingCount = 1;
                state.mVertexBindings[0] = {0, mesh.mVertexStride, false};
                state.mVertexAttributes[attributeCount++] = {0, 0, rhi::Format::kR32G32B32Float, mesh.mPositionOffset};
                if (mesh.HasNormals()) {
                    state.mVertexAttributes[attributeCount++] = {1, 0, rhi::Format::kR32G32B32Float, mesh.mNormalOffset};
                }
                if (mesh.HasUvs()) {
                    state.mVertexAttributes[attributeCount++] = {2, 0, rhi::Format::kR32G32Float, mesh.mUvOffset};
                }
                if (mesh.HasColors()) {
                    state.mVertexAttributes[attributeCount++] = {3, 0, rhi::Format::kR8G8B8A8Unorm, mesh.mColorOffset};
                }
            }

            if (cmd.mInstance.mBuffer != nullptr && cmd.mInstance.mAttributeCount > 0) {
                state.mVertexBindingCount = 2;
                state.mVertexBindings[1] = {1, cmd.mInstance.mStride, true};
                for (uint32_t i = 0; i < cmd.mInstance.mAttributeCount; ++i) {
                    const InstanceAttribute& attr = cmd.mInstance.mAttributes[i];
                    state.mVertexAttributes[attributeCount++] = {attr.mLocation, 1, attr.mFormat, attr.mOffset};
                }
            }

            state.mVertexAttributeCount = attributeCount;
            return state;
        }

        // Shared queueing tail for Draw/DrawFullscreen: snapshot the frame
        // state into the command, resolve the pipeline, enqueue.
        static void QueueDraw(Impl& impl, DrawCmd& cmd) {
            if (impl.mDraws.size() >= kMaxDrawsPerFrame) {
                impl.mLastError = "Draw: per-frame draw limit exceeded";
                return;
            }
            rhi::GraphicsPipeline pipeline;
            const rhi::GraphicsPipelineState state = impl.BuildPipelineState(cmd);
            if (!impl.mDevice->GetOrCreateGraphicsPipeline(state, pipeline)) {
                impl.mLastError = "Draw: pipeline: " + impl.mDevice->GetLastError();
                return;
            }
            cmd.mPipelineHash = state.GetHash();
            impl.mDraws.push_back(std::move(cmd));
        }
    };

    Renderer::Renderer() : mImpl(std::make_unique<Impl>()) {}
    Renderer::~Renderer() = default;

    bool Renderer::Init(rhi::Device& device, rhi::DefaultPipelineCache& cache,
            uint32_t width, uint32_t height, std::string& error) {
        if (mImpl->mDevice != nullptr) {
            error = "Renderer already initialized";
            return false;
        }
        mImpl->mDevice = &device;
        mImpl->mCache = &cache;
        mImpl->mWidth = width;
        mImpl->mHeight = height;

        rhi::ImageCreateInfo depthInfo{};
        depthInfo.mType = rhi::ImageType::k2D;
        depthInfo.mWidth = width;
        depthInfo.mHeight = height;
        depthInfo.mFormat = rhi::Format::kD32Float;
        depthInfo.mUsage = rhi::ImageUsage::kDepthAttachment;
        if (!device.CreateImage(depthInfo, mImpl->mDepthImage)) {
            error = "Renderer: depth image: " + device.GetLastError();
            mImpl->mDevice = nullptr;
            return false;
        }
        mImpl->mStatsTime = std::chrono::steady_clock::now();
        moe::Logger::info("Renderer initialized ({}x{})", width, height);
        return true;
    }

    void Renderer::Destroy() {
        if (mImpl == nullptr || mImpl->mDevice == nullptr) {
            return;
        }
        for (auto& set : mImpl->mFrameSets) {
            set->Destroy();
        }
        mImpl->mFrameSets.clear();
        // destroy live render targets before the cache clears the slots
        mImpl->mTargets.ForEach([](RenderTarget& target) {
            if (target.mDepthImage) {
                target.mDepthImage->Destroy();
            }
            if (target.mImage) {
                target.mImage->Destroy();
            }
        });
        mImpl->mTargets.Clear();
        mImpl->mDepthImage.Destroy();
        mImpl->mDevice = nullptr;
        moe::Logger::info("Renderer destroyed");
    }

    void Renderer::BeginFrame(rhi::CommandList& cmd, const rhi::Image& swapchainImage,
            rhi::Format swapchainFormat, const float clearColor[4]) {
        for (auto& set : mImpl->mFrameSets) {
            set->Destroy();
        }
        mImpl->mFrameSets.clear();
        mImpl->mCmd = &cmd;
        mImpl->mSwapchainImage = &swapchainImage;
        mImpl->mSwapchainFormat = swapchainFormat;
        std::memcpy(mImpl->mClearColor, clearColor, sizeof(float) * 4);
        mImpl->mDraws.clear();
        mImpl->mState = DrawState{};
        mImpl->mTarget = {};
        mImpl->mTargetPtr = nullptr;
        mImpl->mImageCount = 0;
        mImpl->mSamplerCount = 0;
        mImpl->mInstance = Impl::InstanceBind{};
        mImpl->mTableSize = 0;
    }

    void Renderer::SetState(const DrawState& state) {
        mImpl->mState = state;
    }

    void Renderer::BindTarget(RenderTargetHandle target) {
        mImpl->mTarget = target;
        mImpl->mTargetPtr = mImpl->ResolveTarget(target);
    }

    void Renderer::BindImage(uint32_t binding, const rhi::Image& image) {
        for (uint32_t i = 0; i < mImpl->mImageCount; ++i) {
            if (mImpl->mImages[i].mBinding == binding) {
                mImpl->mImages[i].mImage = &image;
                return;
            }
        }
        if (mImpl->mImageCount >= kMaxTextureBindings) {
            mImpl->mLastError = "BindImage: per-frame binding limit exceeded";
            return;
        }
        mImpl->mImages[mImpl->mImageCount++] = {binding, &image};
    }

    void Renderer::BindSampler(uint32_t binding, const rhi::Sampler& sampler) {
        for (uint32_t i = 0; i < mImpl->mSamplerCount; ++i) {
            if (mImpl->mSamplers[i].mBinding == binding) {
                mImpl->mSamplers[i].mSampler = &sampler;
                return;
            }
        }
        if (mImpl->mSamplerCount >= kMaxTextureBindings) {
            mImpl->mLastError = "BindSampler: per-frame binding limit exceeded";
            return;
        }
        mImpl->mSamplers[mImpl->mSamplerCount++] = {binding, &sampler};
    }

    void Renderer::BindInstanceBuffer(const rhi::Buffer& buffer, uint32_t stride,
            const InstanceAttribute* attributes, uint32_t attributeCount) {
        if (attributeCount > kMaxInstanceAttributes) {
            mImpl->mLastError = "BindInstanceBuffer: too many attributes";
            return;
        }
        mImpl->mInstance.mBuffer = &buffer;
        mImpl->mInstance.mStride = stride;
        mImpl->mInstance.mAttributeCount = attributeCount;
        for (uint32_t i = 0; i < attributeCount; ++i) {
            mImpl->mInstance.mAttributes[i] = attributes[i];
        }
    }

    int32_t Renderer::GetPushConstant(const rhi::ShaderProgram& program, const char* name) const {
        return mImpl->LookupField(program, name);
    }

    bool Renderer::SetPushConstant(int32_t index, const void* data, size_t size) {
        if (index < 0 || static_cast<size_t>(index) >= mImpl->mFields.size()) {
            mImpl->mLastError = "SetPushConstant: invalid index";
            return false;
        }
        const auto& field = mImpl->mFields[static_cast<size_t>(index)];
        if (size != field.mSize) {
            mImpl->mLastError = "SetPushConstant: size mismatch (shader expects " +
                    std::to_string(field.mSize) + ", got " + std::to_string(size) + ")";
            return false;
        }
        if (field.mOffset + field.mSize > kMaxPushConstantBytes) {
            mImpl->mLastError = "SetPushConstant: exceeds the value table";
            return false;
        }
        std::memcpy(mImpl->mTable.data() + field.mOffset, data, size);
        mImpl->mTableSize = std::max(mImpl->mTableSize, field.mOffset + static_cast<uint32_t>(size));
        return true;
    }

    RenderTargetHandle Renderer::CreateRenderTarget(uint32_t width, uint32_t height,
            rhi::Format format, bool withDepth, std::string& error) {
        if (width == 0 || height == 0) {
            error = "CreateRenderTarget: zero size";
            return {};
        }
        RenderTarget target;
        target.mWidth = width;
        target.mHeight = height;
        target.mFormat = format;
        target.mHasDepth = withDepth;
        target.mImage = std::make_unique<rhi::Image>();
        target.mDepthImage = withDepth ? std::make_unique<rhi::Image>() : nullptr;

        rhi::ImageCreateInfo colorInfo{};
        colorInfo.mType = rhi::ImageType::k2D;
        colorInfo.mWidth = width;
        colorInfo.mHeight = height;
        colorInfo.mFormat = format;
        colorInfo.mUsage = rhi::ImageUsage::kColorAttachment | rhi::ImageUsage::kSampled;
        if (!mImpl->mDevice->CreateImage(colorInfo, *target.mImage)) {
            error = "CreateRenderTarget: color image: " + mImpl->mDevice->GetLastError();
            return {};
        }
        if (withDepth) {
            rhi::ImageCreateInfo depthInfo{};
            depthInfo.mType = rhi::ImageType::k2D;
            depthInfo.mWidth = width;
            depthInfo.mHeight = height;
            depthInfo.mFormat = rhi::Format::kD32Float;
            depthInfo.mUsage = rhi::ImageUsage::kDepthAttachment;
            if (!mImpl->mDevice->CreateImage(depthInfo, *target.mDepthImage)) {
                error = "CreateRenderTarget: depth image: " + mImpl->mDevice->GetLastError();
                target.mImage->Destroy();
                return {};
            }
        }
        const RenderTargetHandle handle = mImpl->mTargets.Add(std::move(target));
        moe::Logger::info("Renderer created render target ({}x{} {})",
                width, height, withDepth ? "depth" : "color-only");
        return handle;
    }

    void Renderer::DestroyRenderTarget(RenderTargetHandle handle) {        RenderTarget* target = mImpl->mTargets.Get(handle);
        if (target == nullptr) {
            mImpl->mLastError = "DestroyRenderTarget: stale handle";
            return;
        }
        const uint32_t width = target->mWidth;
        const uint32_t height = target->mHeight;
        const bool hasDepth = target->mHasDepth;
        if (target->mDepthImage) {
            target->mDepthImage->Destroy();
        }
        target->mImage->Destroy();
        mImpl->mTargets.Remove(handle);
        moe::Logger::info("Renderer destroyed render target ({}x{} {})",
                width, height, hasDepth ? "depth" : "color-only");
    }

    RenderTarget* Renderer::GetRenderTarget(RenderTargetHandle handle) {
        return mImpl->mTargets.Get(handle);
    }

    void Renderer::Draw(const UploadedMesh& mesh, const rhi::ShaderProgram& program,
            rhi::PrimitiveTopology topology, uint32_t instanceCount) {
        Impl::DrawCmd cmd;
        cmd.mProgram = &program;
        cmd.mMesh = &mesh;
        cmd.mTopology = topology;
        cmd.mInstanceCount = instanceCount;
        cmd.mState = mImpl->mState;
        cmd.mTarget = mImpl->mTarget;
        cmd.mTargetPtr = mImpl->mTargetPtr;
        cmd.mImages = mImpl->mImages;
        cmd.mImageCount = mImpl->mImageCount;
        cmd.mSamplers = mImpl->mSamplers;
        cmd.mSamplerCount = mImpl->mSamplerCount;
        cmd.mInstance = mImpl->mInstance;
        cmd.mTable = mImpl->mTable;
        cmd.mTableSize = mImpl->mTableSize;
        Impl::QueueDraw(*mImpl, cmd);
    }

    void Renderer::DrawFullscreen(const rhi::ShaderProgram& program) {
        Impl::DrawCmd cmd;
        cmd.mProgram = &program;
        cmd.mState = mImpl->mState;
        cmd.mTarget = mImpl->mTarget;
        cmd.mTargetPtr = mImpl->mTargetPtr;
        cmd.mImages = mImpl->mImages;
        cmd.mImageCount = mImpl->mImageCount;
        cmd.mSamplers = mImpl->mSamplers;
        cmd.mSamplerCount = mImpl->mSamplerCount;
        cmd.mTable = mImpl->mTable;
        cmd.mTableSize = mImpl->mTableSize;
        Impl::QueueDraw(*mImpl, cmd);
    }

    void Renderer::EndFrame() {
        rhi::CommandList& cmd = *mImpl->mCmd;

        // swapchain: PresentSrc (left by App's EndRendering) -> color attachment
        rhi::SyncInfo sync{};
        sync.mSrcStage = rhi::PipelineStage::kBottomOfPipe;
        sync.mSrcAccess = rhi::Access::kNone;
        sync.mDstStage = rhi::PipelineStage::kColorAttachmentOutput;
        sync.mDstAccess = rhi::Access::kColorAttachmentWrite;
        cmd.ImageBarrier(*mImpl->mSwapchainImage, rhi::ImageLayout::kPresentSrc,
                rhi::ImageLayout::kColorAttachment, sync);

        // depth: first frame transitions undefined -> depth attachment
        if (!mImpl->mDepthReady) {
            sync.mSrcStage = rhi::PipelineStage::kTopOfPipe;
            sync.mSrcAccess = rhi::Access::kNone;
            sync.mDstStage = rhi::PipelineStage::kEarlyFragmentTests;
            sync.mDstAccess = rhi::Access::kDepthStencilAttachmentWrite;
            cmd.ImageBarrier(mImpl->mDepthImage, rhi::ImageLayout::kUndefined,
                    rhi::ImageLayout::kDepthStencilAttachment, sync);
            mImpl->mDepthReady = true;
        }

        // sort: render target first (fewer pass switches), then pipeline hash
        auto targetKey = [](const Impl::DrawCmd& d) {
            return d.mTarget.IsValid() ? d.mTarget.mIndex : UINT32_MAX;
        };
        std::stable_sort(mImpl->mDraws.begin(), mImpl->mDraws.end(),
                [&targetKey](const Impl::DrawCmd& a, const Impl::DrawCmd& b) {
                    const uint32_t ta = targetKey(a);
                    const uint32_t tb = targetKey(b);
                    if (ta != tb) {
                        return ta < tb;
                    }
                    return a.mPipelineHash < b.mPipelineHash;
                });

        RenderTarget* activeTarget = nullptr;
        bool passOpen = false;
        for (const auto& draw : mImpl->mDraws) {
            if (!passOpen || draw.mTargetPtr != activeTarget) {
                if (passOpen) {
                    cmd.EndRendering();
                    passOpen = false;
                }
                activeTarget = draw.mTargetPtr;

                // --- pass-external barriers ---
                // 1) sampled textures: any render target being read moves
                //    to ShaderReadOnly (a barrier right before the pass waits
                //    for the previous pass that wrote it)
                for (uint32_t i = 0; i < draw.mImageCount; ++i) {
                    const rhi::Image* tex = draw.mImages[i].mImage;
                    RenderTarget* owner = nullptr;
                    mImpl->mTargets.ForEach([&](RenderTarget& t) {
                        if (owner == nullptr && t.mImage.get() == tex) {
                            owner = &t;
                        }
                    });
                    if (owner == nullptr || owner->mColorLayout == rhi::ImageLayout::kShaderReadOnly) {
                        continue;
                    }
                    sync.mSrcStage = owner->mColorLayout == rhi::ImageLayout::kColorAttachment
                            ? rhi::PipelineStage::kColorAttachmentOutput
                            : rhi::PipelineStage::kTopOfPipe;
                    sync.mSrcAccess = owner->mColorLayout == rhi::ImageLayout::kColorAttachment
                            ? rhi::Access::kColorAttachmentWrite
                            : rhi::Access::kNone;
                    sync.mDstStage = rhi::PipelineStage::kFragmentShader;
                    sync.mDstAccess = rhi::Access::kShaderRead;
                    cmd.ImageBarrier(*owner->mImage, owner->mColorLayout,
                            rhi::ImageLayout::kShaderReadOnly, sync);
                    owner->mColorLayout = rhi::ImageLayout::kShaderReadOnly;
                }

                // 2) the target itself must be in ColorAttachment
                if (activeTarget != nullptr) {
                    if (activeTarget->mColorLayout != rhi::ImageLayout::kColorAttachment) {
                        sync.mSrcStage = activeTarget->mColorLayout == rhi::ImageLayout::kShaderReadOnly
                                ? rhi::PipelineStage::kFragmentShader
                                : rhi::PipelineStage::kTopOfPipe;
                        sync.mSrcAccess = activeTarget->mColorLayout == rhi::ImageLayout::kShaderReadOnly
                                ? rhi::Access::kShaderRead
                                : rhi::Access::kNone;
                        sync.mDstStage = rhi::PipelineStage::kColorAttachmentOutput;
                        sync.mDstAccess = rhi::Access::kColorAttachmentWrite;
                        cmd.ImageBarrier(*activeTarget->mImage, activeTarget->mColorLayout,
                                rhi::ImageLayout::kColorAttachment, sync);
                        activeTarget->mColorLayout = rhi::ImageLayout::kColorAttachment;
                    }
                    if (activeTarget->mHasDepth && activeTarget->mDepthLayout == rhi::ImageLayout::kUndefined) {
                        sync.mSrcStage = rhi::PipelineStage::kTopOfPipe;
                        sync.mSrcAccess = rhi::Access::kNone;
                        sync.mDstStage = rhi::PipelineStage::kEarlyFragmentTests;
                        sync.mDstAccess = rhi::Access::kDepthStencilAttachmentWrite;
                        cmd.ImageBarrier(*activeTarget->mDepthImage, rhi::ImageLayout::kUndefined,
                                rhi::ImageLayout::kDepthStencilAttachment, sync);
                        activeTarget->mDepthLayout = rhi::ImageLayout::kDepthStencilAttachment;
                    }
                }

                // --- begin the pass ---
                if (activeTarget != nullptr) {
                    cmd.BeginRendering(*activeTarget->mImage, mImpl->mClearColor,
                            activeTarget->mHasDepth ? activeTarget->mDepthImage.get() : nullptr,
                            1.0f, rhi::LoadOp::kClear);
                    cmd.SetViewport(activeTarget->mWidth, activeTarget->mHeight);
                } else {
                    cmd.BeginRendering(*mImpl->mSwapchainImage, mImpl->mClearColor,
                            &mImpl->mDepthImage, 1.0f, rhi::LoadOp::kClear);
                    cmd.SetViewport(mImpl->mWidth, mImpl->mHeight);
                }
                passOpen = true;
            }

            // --- per-draw recording ---
            rhi::GraphicsPipeline pipeline;
            const rhi::GraphicsPipelineState state = mImpl->BuildPipelineState(draw);
            if (!mImpl->mDevice->GetOrCreateGraphicsPipeline(state, pipeline)) {
                mImpl->mLastError = "EndFrame: pipeline: " + mImpl->mDevice->GetLastError();
                continue;
            }
            cmd.BindGraphicsPipeline(pipeline);

            // push constants: per declared range of each stage
            const rhi::ShaderStage stages[3] = {
                    rhi::ShaderStage::kVertex, rhi::ShaderStage::kFragment, rhi::ShaderStage::kGeometry};
            for (const rhi::ShaderStage stage : stages) {
                const rhi::Shader* shader = draw.mProgram->GetStage(stage);
                if (shader == nullptr) {
                    continue;
                }
                for (const auto& range : shader->GetReflection().mPushConstantRanges) {
                    const uint32_t end = range.mOffset + range.mSize;
                    if (end > draw.mTableSize) {
                        continue;
                    }
                    cmd.SetPushConstants(pipeline, range.mOffset, range.mSize,
                            draw.mTable.data() + range.mOffset);
                }
            }

            // images/samplers: one descriptor set per draw, bindings exactly
            // as the user bound them
            if (draw.mImageCount > 0 || draw.mSamplerCount > 0) {
                rhi::DescriptorSetLayout layout;
                if (pipeline.GetDescriptorSetLayout(0, layout)) {
                    auto set = std::make_unique<rhi::DescriptorSet>();
                    if (mImpl->mDevice->CreateDescriptorSet(layout, *set)) {
                        bool written = true;
                        for (uint32_t i = 0; i < draw.mImageCount; ++i) {
                            written &= set->WriteImage(draw.mImages[i].mBinding,
                                    *draw.mImages[i].mImage, rhi::DescriptorType::kSampledImage);
                        }
                        for (uint32_t i = 0; i < draw.mSamplerCount; ++i) {
                            written &= set->WriteSampler(draw.mSamplers[i].mBinding,
                                    *draw.mSamplers[i].mSampler);
                        }
                        if (written) {
                            cmd.BindDescriptorSet(pipeline, *set, 0);
                            mImpl->mFrameSets.push_back(std::move(set));
                        } else {
                            mImpl->mLastError = "EndFrame: descriptor write failed "
                                    "(binding not declared in the shader?)";
                            set->Destroy();
                        }
                    }
                }
            }

            if (draw.mMesh != nullptr) {
                const UploadedMesh& mesh = *draw.mMesh;
                cmd.BindVertexBuffer(mesh.mVertexBuffer, 0);
                if (draw.mInstance.mBuffer != nullptr) {
                    cmd.BindVertexBuffer(*draw.mInstance.mBuffer, 1);
                }
                cmd.BindIndexBuffer(mesh.mIndexBuffer);
                cmd.DrawIndexed(mesh.mIndexCount, draw.mInstanceCount, 0, 0, 0);
            } else {
                cmd.Draw(3, 1, 0, 0);
            }
        }
        if (passOpen) {
            cmd.EndRendering();
        }

        // back to PresentSrc for the App's ImGui composite
        sync.mSrcStage = rhi::PipelineStage::kColorAttachmentOutput;
        sync.mSrcAccess = rhi::Access::kColorAttachmentWrite;
        sync.mDstStage = rhi::PipelineStage::kBottomOfPipe;
        sync.mDstAccess = rhi::Access::kNone;
        cmd.ImageBarrier(*mImpl->mSwapchainImage, rhi::ImageLayout::kColorAttachment,
                rhi::ImageLayout::kPresentSrc, sync);

        // per-second stats (avoids log flooding while keeping visibility)
        mImpl->mStatsFrames += 1;
        mImpl->mStatsDraws += mImpl->mDraws.size();
        mImpl->mStatsLastDraws = static_cast<uint32_t>(mImpl->mDraws.size());
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - mImpl->mStatsTime).count();
        if (elapsed >= 1.0) {
            const double fps = static_cast<double>(mImpl->mStatsFrames) / elapsed;
            const double avgDraws = static_cast<double>(mImpl->mStatsDraws) / mImpl->mStatsFrames;
            moe::Logger::info("Renderer frame stats: {:.1f} fps, {:.0f} avg draws/frame, "
                    "{} cached pipelines",
                    fps, avgDraws, mImpl->mCache->GetNodeCount());
            mImpl->mStatsTime = now;
            mImpl->mStatsFrames = 0;
            mImpl->mStatsDraws = 0;
        }

        mImpl->mDraws.clear();
    }

    const std::string& Renderer::GetLastError() const {
        return mImpl->mLastError;
    }
}// namespace moe::neo
