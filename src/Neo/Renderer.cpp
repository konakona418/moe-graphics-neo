#include "Neo/Renderer.hpp"

#include <Core/Error.hpp>
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
        uint32_t mSampleCount{1};

        // push constant name registry (lookup-once, then set by index). Each
        // field keeps its own value: different programs may place fields at
        // the same push constant offset, so values must not share a byte
        // space. Replay writes per field (per declared range of the program).
        struct FieldReg {
            const rhi::ShaderProgram* mProgram{nullptr};
            std::string mName;
            uint32_t mOffset{0};
            uint32_t mSize{0};
            bool mDirty{false};
            std::vector<uint8_t> mValue;
        };
        mutable std::vector<FieldReg> mFields;

        // True when any field of `program` changed since it was last pushed
        // to the command buffer; such fields force a (cheap) pipeline rebind
        // so their values are recorded for the next draw.
        bool HasDirtyFields(const rhi::ShaderProgram& program) const {
            for (const auto& field : mFields) {
                if (field.mProgram == &program && field.mDirty) {
                    return true;
                }
            }
            return false;
        }

        // ---- per-frame GL-style state (reset in BeginFrame; drawn commands
        // are recorded immediately, so later changes never affect them) ----

        struct ImageBinding {
            uint32_t mBinding{0};
            const rhi::Image* mImage{nullptr};
        };
        struct SamplerBinding {
            uint32_t mBinding{0};
            const rhi::Sampler* mSampler{nullptr};
        };
        struct BufferBinding {
            uint32_t mBinding{0};
            const rhi::Buffer* mBuffer{nullptr};
        };
        struct InstanceBind {
            const rhi::Buffer* mBuffer{nullptr};
            uint32_t mStride{0};
            std::array<InstanceAttribute, kMaxInstanceAttributes> mAttributes{};
            uint32_t mAttributeCount{0};
        };

        DrawState mState;
        Camera mCamera;
        bool mCameraSet{false};
        RenderTargetHandle mTarget;
        RenderTarget* mTargetPtr{nullptr};
        std::array<ImageBinding, kMaxTextureBindings> mImages{};
        uint32_t mImageCount{0};
        std::array<SamplerBinding, kMaxTextureBindings> mSamplers{};
        uint32_t mSamplerCount{0};
        std::array<BufferBinding, kMaxTextureBindings> mBuffers{};
        uint32_t mBufferCount{0};
        InstanceBind mInstance;
        // (the per-frame value table was replaced by per-field values:
        // different programs may reuse the same push constant offsets)

        // frame state
        rhi::CommandList* mCmd{nullptr};
        // Borrowed swapchain image: owned by the caller's SwapchainImage
        // object, valid from BeginFrame to EndFrame (both inside the same App
        // callback scope).
        const rhi::Image* mSwapchainImage{nullptr};
        // Owning swapchain of the current frame; its BeginRendering /
        // EndRendering pair manages the swapchain image's layout state.
        rhi::Swapchain* mSwapchain{nullptr};
        rhi::Format mSwapchainFormat{rhi::Format::kUndefined};
        uint32_t mFrameWidth{0};
        uint32_t mFrameHeight{0};
        float mClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        // main (swapchain) depth: tracked layout, transitioned once then kept
        // as an attachment across frames
        rhi::ImageLayout mDepthLayout{rhi::ImageLayout::kUndefined};

        // immediate-mode tracking
        bool mPassOpen{false};
        const char* mPassName{nullptr};
        RenderTarget* mCurrentTarget{nullptr};
        // depth attachment owner of the open pass (null = the renderer's main
        // depth, or no depth attachment); EndPass uses it for the deferred
        // sampling transition
        RenderTarget* mDepthTargetPtr{nullptr};
        uint64_t mCurrentPipelineHash{0};

        Cache<RenderTarget> mTargets;

        // Descriptor sets created during recording; freed at the next
        // BeginFrame (queue is serial, so in-flight usage has finished).
        std::vector<std::unique_ptr<rhi::DescriptorSet>> mFrameSets;

        // per-second frame statistics
        std::chrono::steady_clock::time_point mStatsTime{};
        uint32_t mStatsFrames{0};
        uint64_t mStatsDraws{0};

        RenderTarget* ResolveTarget(RenderTargetHandle handle) {
            return handle.IsValid() ? mTargets.Get(handle) : nullptr;
        }

        // ---- attachment transitions (owner-tracked layout + sync) ----

        // Transitions a target's color image to ColorAttachment. The tracked
        // layout is the barrier source, so chained passes stay synchronized.
        // Multisampled attachments stay in ColorAttachment between passes, so
        // an already-attached image still gets a dependency barrier.
        void EnsureColorAttachment(RenderTarget& target) {
            const bool already = target.mColorLayout == rhi::ImageLayout::kColorAttachment;
            if (already && target.mSampleCount == 1) {
                return;
            }
            rhi::SyncInfo sync{};
            if (already) {
                sync.mSrcStage = rhi::PipelineStage::kColorAttachmentOutput;
                sync.mSrcAccess = rhi::Access::kColorAttachmentWrite;
            } else if (target.mColorLayout == rhi::ImageLayout::kShaderReadOnly) {
                sync.mSrcStage = rhi::PipelineStage::kFragmentShader;
                sync.mSrcAccess = rhi::Access::kShaderRead;
            } else {
                sync.mSrcStage = rhi::PipelineStage::kTopOfPipe;
                sync.mSrcAccess = rhi::Access::kNone;
            }
            sync.mDstStage = rhi::PipelineStage::kColorAttachmentOutput;
            sync.mDstAccess = rhi::Access::kColorAttachmentWrite;
            mCmd->ImageBarrier(target.AttachmentImage(), target.mColorLayout,
                    rhi::ImageLayout::kColorAttachment, sync);
            target.mColorLayout = rhi::ImageLayout::kColorAttachment;
        }

        // Transitions a target's resolve image (mImage) to ColorAttachment
        // before a multisampled pass resolves into it.
        void EnsureResolveAttachment(RenderTarget& target) {
            if (target.mSampleCount == 1) {
                return;
            }
            const bool already = target.mResolveLayout == rhi::ImageLayout::kColorAttachment;
            if (already) {
                return;
            }
            rhi::SyncInfo sync{};
            sync.mSrcStage = target.mResolveLayout == rhi::ImageLayout::kShaderReadOnly
                    ? rhi::PipelineStage::kFragmentShader
                    : rhi::PipelineStage::kTopOfPipe;
            sync.mSrcAccess = target.mResolveLayout == rhi::ImageLayout::kShaderReadOnly
                    ? rhi::Access::kShaderRead
                    : rhi::Access::kNone;
            sync.mDstStage = rhi::PipelineStage::kColorAttachmentOutput;
            sync.mDstAccess = rhi::Access::kColorAttachmentWrite;
            mCmd->ImageBarrier(*target.mImage, target.mResolveLayout,
                    rhi::ImageLayout::kColorAttachment, sync);
            target.mResolveLayout = rhi::ImageLayout::kColorAttachment;
        }

        // Same for a target's depth image (no-op when the target has none).
        void EnsureDepthAttachment(RenderTarget& target) {
            if (!target.mHasDepth || target.mDepthLayout == rhi::ImageLayout::kDepthStencilAttachment) {
                return;
            }
            rhi::SyncInfo sync{};
            sync.mSrcStage = target.mDepthLayout == rhi::ImageLayout::kShaderReadOnly
                    ? rhi::PipelineStage::kFragmentShader
                    : rhi::PipelineStage::kTopOfPipe;
            sync.mSrcAccess = target.mDepthLayout == rhi::ImageLayout::kShaderReadOnly
                    ? rhi::Access::kDepthStencilAttachmentRead
                    : rhi::Access::kNone;
            sync.mDstStage = rhi::PipelineStage::kEarlyFragmentTests;
            sync.mDstAccess = rhi::Access::kDepthStencilAttachmentWrite;
            mCmd->ImageBarrier(*target.mDepthImage, target.mDepthLayout,
                    rhi::ImageLayout::kDepthStencilAttachment, sync);
            target.mDepthLayout = rhi::ImageLayout::kDepthStencilAttachment;
        }

        // Same for the renderer's own main depth (swapchain passes).
        void EnsureMainDepthAttachment() {
            if (mDepthLayout == rhi::ImageLayout::kDepthStencilAttachment) {
                return;
            }
            rhi::SyncInfo sync{};
            sync.mSrcStage = rhi::PipelineStage::kTopOfPipe;
            sync.mSrcAccess = rhi::Access::kNone;
            sync.mDstStage = rhi::PipelineStage::kEarlyFragmentTests;
            sync.mDstAccess = rhi::Access::kDepthStencilAttachmentWrite;
            mCmd->ImageBarrier(mDepthImage, mDepthLayout,
                    rhi::ImageLayout::kDepthStencilAttachment, sync);
            mDepthLayout = rhi::ImageLayout::kDepthStencilAttachment;
        }

        // Returns a target's depth image to ShaderReadOnly after a pass, so
        // later passes can sample it (deferred sampling transition).
        void ReleaseDepthAttachment(RenderTarget& target) {
            if (!target.mHasDepth || target.mDepthLayout == rhi::ImageLayout::kShaderReadOnly) {
                return;
            }
            rhi::SyncInfo sync{};
            sync.mSrcStage = rhi::PipelineStage::kEarlyFragmentTests;
            sync.mSrcAccess = rhi::Access::kDepthStencilAttachmentWrite;
            sync.mDstStage = rhi::PipelineStage::kFragmentShader;
            sync.mDstAccess = rhi::Access::kDepthStencilAttachmentRead;
            mCmd->ImageBarrier(*target.mDepthImage, target.mDepthLayout,
                    rhi::ImageLayout::kShaderReadOnly, sync);
            target.mDepthLayout = rhi::ImageLayout::kShaderReadOnly;
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

        // Builds the RHI pipeline state for the current frame state: vertex
        // layout from the uploaded mesh (binding 0) + explicit instance
        // attributes (binding 1, only when instanced), state from mState,
        // formats from the target.
        rhi::GraphicsPipelineState BuildPipelineState(const UploadedMesh* mesh,
                const rhi::ShaderProgram& program, rhi::PrimitiveTopology topology,
                bool useInstancing) {
            rhi::GraphicsPipelineState state{};
            state.mProgram = &program;
            state.mTopology = topology;

            state.mColorFormatCount = 1;
            state.mColorFormats[0] =
                    mTargetPtr != nullptr ? mTargetPtr->mFormat : mSwapchainFormat;
            state.mDepthFormat = rhi::Format::kD32Float;
            state.mMultisample.mSampleCount = static_cast<uint8_t>(
                    mTargetPtr != nullptr ? mTargetPtr->mSampleCount : mSampleCount);

            state.mBlendAttachmentCount = 1;
            state.mBlendAttachments[0].mBlendEnabled = mState.mBlendEnabled;
            state.mBlendAttachments[0].mSrcColor = mState.mBlendSrcColor;
            state.mBlendAttachments[0].mDstColor = mState.mBlendDstColor;
            state.mBlendAttachments[0].mColorOp = mState.mBlendColorOp;
            state.mBlendAttachments[0].mSrcAlpha = mState.mBlendSrcAlpha;
            state.mBlendAttachments[0].mDstAlpha = mState.mBlendDstAlpha;
            state.mBlendAttachments[0].mAlphaOp = mState.mBlendAlphaOp;

            state.mRaster.mCullMode = mState.mCullMode;
            state.mRaster.mFrontFace = mState.mFrontFace;
            state.mRaster.mPolygonMode = mState.mPolygonMode;
            if (mesh == nullptr) {
                // Fullscreen passes: the fullscreen triangle's winding is
                // fixed by the generated UV formula (clockwise in window
                // space), so culling would always discard it. Ignore it.
                state.mRaster.mCullMode = rhi::CullMode::kNone;
            }

            state.mDepth.mTestEnable = mState.mDepthTest;
            state.mDepth.mWriteEnable = mState.mDepthWrite;
            state.mDepth.mCompareOp = mState.mDepthCompareOp;

            uint32_t attributeCount = 0;
            if (mesh != nullptr) {
                state.mVertexBindingCount = 1;
                state.mVertexBindings[0] = {0, mesh->mVertexStride, false};
                // attribute locations are assigned by channel presence, in
                // shader-idiomatic order (position, normal, uv, color) — a
                // mesh without normals puts uv at location 1, etc. (The
                // location is captured before the index increments: the
                // evaluation order of `arr[i++] = {i, ...}` is not portable.)
                const uint32_t positionLocation = attributeCount;
                state.mVertexAttributes[attributeCount++] = {positionLocation, 0, rhi::Format::kR32G32B32Float, mesh->mPositionOffset};
                if (mesh->HasNormals()) {
                    const uint32_t normalLocation = attributeCount;
                    state.mVertexAttributes[attributeCount++] = {normalLocation, 0, rhi::Format::kR32G32B32Float, mesh->mNormalOffset};
                }
                if (mesh->HasUvs()) {
                    const uint32_t uvLocation = attributeCount;
                    state.mVertexAttributes[attributeCount++] = {uvLocation, 0, rhi::Format::kR32G32Float, mesh->mUvOffset};
                }
                if (mesh->HasColors()) {
                    const uint32_t colorLocation = attributeCount;
                    state.mVertexAttributes[attributeCount++] = {colorLocation, 0, rhi::Format::kR8G8B8A8Unorm, mesh->mColorOffset};
                }
            }

            if (useInstancing && mInstance.mBuffer != nullptr && mInstance.mAttributeCount > 0) {
                state.mVertexBindingCount = 2;
                state.mVertexBindings[1] = {1, mInstance.mStride, true};
                for (uint32_t i = 0; i < mInstance.mAttributeCount; ++i) {
                    const InstanceAttribute& attr = mInstance.mAttributes[i];
                    state.mVertexAttributes[attributeCount++] = {attr.mLocation, 1, attr.mFormat, attr.mOffset};
                }
            }

            state.mVertexAttributeCount = attributeCount;
            return state;
        }
    };

    Renderer::Renderer() : mImpl(std::make_unique<Impl>()) {}
    Renderer::~Renderer() = default;

    bool Renderer::Init(rhi::Device& device, rhi::DefaultPipelineCache& cache,
            uint32_t width, uint32_t height, uint32_t sampleCount) {
        if (mImpl->mDevice != nullptr) {
            return moe::Fail("Renderer already initialized");
        }
        if (sampleCount != 1 && sampleCount != 2 && sampleCount != 4 && sampleCount != 8) {
            return moe::Fail("Renderer: sample count must be 1, 2, 4 or 8");
        }
        if (sampleCount > device.GetMaxSampleCount()) {
            return moe::Fail("Renderer: sample count exceeds device support");
        }
        mImpl->mDevice = &device;
        mImpl->mCache = &cache;
        mImpl->mWidth = width;
        mImpl->mHeight = height;
        mImpl->mSampleCount = sampleCount;

        rhi::ImageCreateInfo depthInfo{};
        depthInfo.mType = rhi::ImageType::k2D;
        depthInfo.mWidth = width;
        depthInfo.mHeight = height;
        depthInfo.mFormat = rhi::Format::kD32Float;
        depthInfo.mUsage = rhi::ImageUsage::kDepthAttachment;
        depthInfo.mSampleCount = sampleCount;
        if (!device.CreateImage(depthInfo, mImpl->mDepthImage)) {
            mImpl->mDevice = nullptr;
            return moe::Fail("Renderer: depth image: " + moe::Error::Get());
        }
        mImpl->mStatsTime = std::chrono::steady_clock::now();
        if (sampleCount > 1) {
            moe::Logger::info("Renderer initialized ({}x{}, {}x MSAA)", width, height, sampleCount);
        } else {
            moe::Logger::info("Renderer initialized ({}x{})", width, height);
        }
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
        mImpl->mTargets.ForEach([](RenderTarget& target) {
            if (target.mDepthImage) {
                target.mDepthImage->Destroy();
            }
            if (target.mMsaaImage) {
                target.mMsaaImage->Destroy();
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

    void Renderer::BeginFrame(rhi::CommandList& cmd, const SwapchainImage& frame,
            const float clearColor[4]) {
        for (auto& set : mImpl->mFrameSets) {
            set->Destroy();
        }
        mImpl->mFrameSets.clear();
        mImpl->mCmd = &cmd;
        mImpl->mSwapchainImage = &frame.GetImage();
        mImpl->mSwapchain = &frame.GetSwapchain();
        mImpl->mSwapchainFormat = frame.GetFormat();
        mImpl->mFrameWidth = frame.GetWidth();
        mImpl->mFrameHeight = frame.GetHeight();
        std::memcpy(mImpl->mClearColor, clearColor, sizeof(float) * 4);
        mImpl->mState = DrawState{};
        mImpl->mCameraSet = false;
        mImpl->mTarget = {};
        mImpl->mTargetPtr = nullptr;
        mImpl->mImageCount = 0;
        mImpl->mSamplerCount = 0;
        mImpl->mBufferCount = 0;
        mImpl->mInstance = Impl::InstanceBind{};
        mImpl->mPassOpen = false;
        mImpl->mPassName = nullptr;
        mImpl->mCurrentTarget = nullptr;
        mImpl->mDepthTargetPtr = nullptr;
        mImpl->mCurrentPipelineHash = 0;
    }

    void Renderer::EndFrame() {
        Impl& impl = *mImpl;

        if (impl.mPassOpen) {
            moe::Logger::warn("Renderer: pass '{}' left open at EndFrame; closing it", impl.mPassName);
            if (impl.mCurrentTarget == nullptr) {
                impl.mSwapchain->EndRendering(*impl.mCmd);
            } else {
                impl.mCmd->EndRendering();
            }
            impl.mPassOpen = false;
            impl.mDepthTargetPtr = nullptr;
        }

        // per-second stats (avoids log flooding while keeping visibility)
        impl.mStatsFrames += 1;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - impl.mStatsTime).count();
        if (elapsed >= 1.0) {
            const double fps = static_cast<double>(impl.mStatsFrames) / elapsed;
            const double avgDraws = static_cast<double>(impl.mStatsDraws) / impl.mStatsFrames;
            moe::Logger::info("Renderer frame stats: {:.1f} fps, {:.0f} avg draws/frame, "
                    "{} cached pipelines",
                    fps, avgDraws, impl.mCache->GetNodeCount());
            impl.mStatsTime = now;
            impl.mStatsFrames = 0;
            impl.mStatsDraws = 0;
        }
    }

    // ---- GL-style state (internal; PassContext is the public entry) ----

    void Renderer::SetStateInternal(const DrawState& state) {
        mImpl->mState = state;
    }

    void Renderer::SetCameraInternal(const Camera& camera) {
        mImpl->mCamera = camera;
        mImpl->mCameraSet = true;
    }

    const Camera* Renderer::GetCameraInternal() const {
        return mImpl->mCameraSet ? &mImpl->mCamera : nullptr;
    }

    void Renderer::ClearTextureBindingsInternal() {
        mImpl->mImageCount = 0;
        mImpl->mSamplerCount = 0;
    }

    void Renderer::BindImageInternal(uint32_t binding, const rhi::Image& image) {
        for (uint32_t i = 0; i < mImpl->mImageCount; ++i) {
            if (mImpl->mImages[i].mBinding == binding) {
                mImpl->mImages[i].mImage = &image;
                return;
            }
        }
        if (mImpl->mImageCount >= kMaxTextureBindings) {
            moe::Error::Set("BindImage: per-frame binding limit exceeded");
            return;
        }
        mImpl->mImages[mImpl->mImageCount++] = {binding, &image};
    }

    void Renderer::BindSamplerInternal(uint32_t binding, const rhi::Sampler& sampler) {
        for (uint32_t i = 0; i < mImpl->mSamplerCount; ++i) {
            if (mImpl->mSamplers[i].mBinding == binding) {
                mImpl->mSamplers[i].mSampler = &sampler;
                return;
            }
        }
        if (mImpl->mSamplerCount >= kMaxTextureBindings) {
            moe::Error::Set("BindSampler: per-frame binding limit exceeded");
            return;
        }
        mImpl->mSamplers[mImpl->mSamplerCount++] = {binding, &sampler};
    }

    void Renderer::BindBufferInternal(uint32_t binding, const rhi::Buffer& buffer) {
        for (uint32_t i = 0; i < mImpl->mBufferCount; ++i) {
            if (mImpl->mBuffers[i].mBinding == binding) {
                mImpl->mBuffers[i].mBuffer = &buffer;
                return;
            }
        }
        if (mImpl->mBufferCount >= kMaxTextureBindings) {
            moe::Error::Set("BindBuffer: per-frame binding limit exceeded");
            return;
        }
        mImpl->mBuffers[mImpl->mBufferCount++] = {binding, &buffer};
    }

    void Renderer::BindInstanceBufferInternal(const rhi::Buffer& buffer, uint32_t stride,
            const InstanceAttribute* attributes, uint32_t attributeCount) {
        if (attributeCount > kMaxInstanceAttributes) {
            moe::Error::Set("BindInstanceBuffer: too many attributes");
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

    uint32_t Renderer::GetPushConstantSize(int32_t index) const {
        if (index < 0 || static_cast<size_t>(index) >= mImpl->mFields.size()) {
            return 0;
        }
        return mImpl->mFields[static_cast<size_t>(index)].mSize;
    }

    bool Renderer::SetPushConstantInternal(int32_t index, const void* data, size_t size) {
        if (index < 0 || static_cast<size_t>(index) >= mImpl->mFields.size()) {
            moe::Error::Set("SetPushConstant: invalid index");
            return false;
        }
        auto& field = mImpl->mFields[static_cast<size_t>(index)];
        if (size != field.mSize) {
            moe::Error::Set("SetPushConstant: size mismatch (shader expects " +
                    std::to_string(field.mSize) + ", got " + std::to_string(size) + ")");
            return false;
        }
        field.mValue.assign(static_cast<const uint8_t*>(data),
                static_cast<const uint8_t*>(data) + size);
        field.mDirty = true;
        return true;
    }

    // ---- engine-recognized barriers (bookkeeping stays in sync) ----

    void Renderer::ImageBarrier(const rhi::Image& image, rhi::ImageLayout srcLayout,
            rhi::ImageLayout dstLayout, const rhi::SyncInfo& sync) {
        if (mImpl->mCmd == nullptr) {
            return;
        }
        mImpl->mCmd->ImageBarrier(image, srcLayout, dstLayout, sync);
        mImpl->mTargets.ForEach([&](RenderTarget& target) {
            if (target.mImage.get() == &image) {
                target.mColorLayout = dstLayout;
            }
            if (target.mDepthImage.get() == &image) {
                target.mDepthLayout = dstLayout;
            }
        });
    }

    void Renderer::BufferBarrier(const rhi::Buffer& buffer, const rhi::SyncInfo& sync) {
        if (mImpl->mCmd != nullptr) {
            mImpl->mCmd->BufferBarrier(buffer, sync);
        }
    }

    void Renderer::MemoryBarrier(const rhi::SyncInfo& sync) {
        if (mImpl->mCmd != nullptr) {
            mImpl->mCmd->MemoryBarrier(sync);
        }
    }

    // ---- render targets ----

    RenderTargetHandle Renderer::CreateRenderTarget(uint32_t width, uint32_t height,
            rhi::Format format, bool withDepth, uint32_t sampleCount) {
        if (width == 0 || height == 0) {
            moe::Error::Set("CreateRenderTarget: zero size");
            return {};
        }
        const uint32_t samples = sampleCount == 0 ? mImpl->mSampleCount : sampleCount;
        if (samples != 1 && samples != 2 && samples != 4 && samples != 8) {
            moe::Error::Set("CreateRenderTarget: sample count must be 1, 2, 4 or 8");
            return {};
        }
        if (samples > mImpl->mDevice->GetMaxSampleCount()) {
            moe::Error::Set("CreateRenderTarget: sample count exceeds device support");
            return {};
        }
        RenderTarget target;
        target.mWidth = width;
        target.mHeight = height;
        target.mFormat = format;
        target.mSampleCount = samples;
        target.mHasDepth = withDepth;
        target.mImage = std::make_unique<rhi::Image>();
        target.mDepthImage = withDepth ? std::make_unique<rhi::Image>() : nullptr;

        rhi::ImageCreateInfo colorInfo{};
        colorInfo.mType = rhi::ImageType::k2D;
        colorInfo.mWidth = width;
        colorInfo.mHeight = height;
        colorInfo.mFormat = format;
        colorInfo.mUsage = rhi::ImageUsage::kColorAttachment | rhi::ImageUsage::kSampled
                | rhi::ImageUsage::kTransferSrc;
        if (!mImpl->mDevice->CreateImage(colorInfo, *target.mImage)) {
            moe::Error::Set("CreateRenderTarget: color image: " + moe::Error::Get());
            return {};
        }
        if (samples > 1) {
            target.mMsaaImage = std::make_unique<rhi::Image>();
            rhi::ImageCreateInfo msaaInfo = colorInfo;
            msaaInfo.mUsage = rhi::ImageUsage::kColorAttachment;
            msaaInfo.mSampleCount = samples;
            if (!mImpl->mDevice->CreateImage(msaaInfo, *target.mMsaaImage)) {
                moe::Error::Set("CreateRenderTarget: multisample image: " + moe::Error::Get());
                target.mImage->Destroy();
                return {};
            }
        }
        if (withDepth) {
            rhi::ImageCreateInfo depthInfo{};
            depthInfo.mType = rhi::ImageType::k2D;
            depthInfo.mWidth = width;
            depthInfo.mHeight = height;
            depthInfo.mFormat = rhi::Format::kD32Float;
            depthInfo.mUsage = rhi::ImageUsage::kDepthAttachment | rhi::ImageUsage::kSampled;
            depthInfo.mSampleCount = samples;
            if (!mImpl->mDevice->CreateImage(depthInfo, *target.mDepthImage)) {
                moe::Error::Set("CreateRenderTarget: depth image: " + moe::Error::Get());
                if (target.mMsaaImage) {
                    target.mMsaaImage->Destroy();
                }
                target.mImage->Destroy();
                return {};
            }
        }
        const RenderTargetHandle handle = mImpl->mTargets.Add(std::move(target));
        moe::Logger::info("Renderer created render target ({}x{} {}{})",
                width, height, withDepth ? "depth" : "color-only",
                samples > 1 ? " msaa" : "");
        return handle;
    }

    void Renderer::DestroyRenderTarget(RenderTargetHandle handle) {
        RenderTarget* target = mImpl->mTargets.Get(handle);
        if (target == nullptr) {
            moe::Error::Set("DestroyRenderTarget: stale handle");
            return;
        }
        const uint32_t width = target->mWidth;
        const uint32_t height = target->mHeight;
        const bool hasDepth = target->mHasDepth;
        if (target->mDepthImage) {
            target->mDepthImage->Destroy();
        }
        if (target->mMsaaImage) {
            target->mMsaaImage->Destroy();
        }
        target->mImage->Destroy();
        mImpl->mTargets.Remove(handle);
        moe::Logger::info("Renderer destroyed render target ({}x{} {})",
                width, height, hasDepth ? "depth" : "color-only");
    }

    RenderTarget* Renderer::GetRenderTarget(RenderTargetHandle handle) {
        return mImpl->mTargets.Get(handle);
    }

    uint32_t Renderer::GetSampleCount() const {
        return mImpl->mSampleCount;
    }

    // ---- passes ----

    void Renderer::BeginPass(const PassDesc& desc) {
        Impl& impl = *mImpl;
        rhi::CommandList& cmd = *impl.mCmd;

        if (impl.mPassOpen) {
            moe::Logger::warn("Renderer: pass '{}' still open when '{}' begins; closing it",
                    impl.mPassName != nullptr ? impl.mPassName : "?", desc.mName);
            cmd.EndRendering();
            impl.mPassOpen = false;
        }

        // ---- resolve attachments (own or borrowed from another target) ----
        const bool swapchainColor = !desc.mColor.mTarget.IsValid();
        RenderTarget* colorTarget = swapchainColor ? nullptr : impl.ResolveTarget(desc.mColor.mTarget);
        if (!swapchainColor && colorTarget == nullptr) {
            moe::Error::Set("BeginPass: color target is stale");
            return;
        }

        RenderTarget* depthTarget = nullptr;
        if (desc.mDepth.mTarget.IsValid()) {
            depthTarget = impl.ResolveTarget(desc.mDepth.mTarget);
            if (depthTarget == nullptr || !depthTarget->mHasDepth) {
                moe::Error::Set("BeginPass: depth target is stale or has no depth");
                return;
            }
        } else if (colorTarget != nullptr && colorTarget->mHasDepth) {
            depthTarget = colorTarget; // the color target's own depth
        }
        // swapchain passes fall back to the renderer's main depth
        const bool mainDepth = depthTarget == nullptr && swapchainColor;

        impl.mCurrentTarget = colorTarget;
        impl.mTarget = desc.mColor.mTarget;
        impl.mTargetPtr = colorTarget;
        impl.mDepthTargetPtr = depthTarget;
        impl.mPassName = desc.mName != nullptr ? desc.mName : "?";
        impl.mCurrentPipelineHash = 0; // pipelines are rebound per pass

        // ---- layout transitions + sync (owner-tracked) ----
        if (colorTarget != nullptr) {
            impl.EnsureColorAttachment(*colorTarget);
            impl.EnsureResolveAttachment(*colorTarget);
        }
        if (depthTarget != nullptr) {
            impl.EnsureDepthAttachment(*depthTarget);
        } else if (mainDepth) {
            impl.EnsureMainDepthAttachment();
        }

        const rhi::Image* depthImage = depthTarget != nullptr ? depthTarget->mDepthImage.get()
                : (mainDepth ? &impl.mDepthImage : nullptr);
        const rhi::LoadOp depthLoadOp = desc.mDepth.mLoadOp;

        if (colorTarget != nullptr) {
            const rhi::Image* resolveImage =
                    colorTarget->mSampleCount > 1 ? colorTarget->mImage.get() : nullptr;
            cmd.BeginRendering(colorTarget->AttachmentImage(), impl.mClearColor, depthImage, 1.0f,
                    desc.mColor.mLoadOp, depthLoadOp, resolveImage);
            cmd.SetViewport(colorTarget->mWidth, colorTarget->mHeight);
        } else {
            // the Swapchain owns its color image's layout state
            impl.mSwapchain->BeginRendering(cmd, impl.mClearColor, desc.mColor.mLoadOp,
                    depthImage, 1.0f, depthLoadOp);
            cmd.SetViewport(impl.mFrameWidth, impl.mFrameHeight);
        }
        impl.mPassOpen = true;
    }

    void Renderer::EndPass(const PassDesc& desc) {
        Impl& impl = *mImpl;
        if (!impl.mPassOpen) {
            moe::Error::Set("EndPass: no active pass");
            return;
        }
        impl.mPassOpen = false;

        if (impl.mCurrentTarget == nullptr) {
            // swapchain pass: Swapchain::EndRendering ends the pass and moves
            // the image to PresentSrc (tracked by the swapchain). A borrowed
            // depth still returns to a sampleable layout.
            impl.mSwapchain->EndRendering(*impl.mCmd);
            if (impl.mDepthTargetPtr != nullptr) {
                impl.ReleaseDepthAttachment(*impl.mDepthTargetPtr);
            }
            impl.mDepthTargetPtr = nullptr;
            return;
        }

        impl.mCmd->EndRendering();

        // Deferred sampling transition (vulkan 1.3 dynamic rendering forbids
        // pipeline barriers inside a render pass): immediately after the pass,
        // move the color attachment (and the depth attachment, own or
        // borrowed) to their read layouts, so later passes can sample them
        // without any barrier. The next pass that draws to one transitions it
        // back in BeginPass.
        {
            RenderTarget* target = impl.mCurrentTarget;
            rhi::CommandList& cmd = *impl.mCmd;
            rhi::SyncInfo sync{};
            if (target->mSampleCount > 1) {
                // multisampled: the attachment stays in ColorAttachment (it is
                // never sampled); only the resolved image is handed on
                if (target->mResolveLayout != rhi::ImageLayout::kShaderReadOnly) {
                    sync.mSrcStage = rhi::PipelineStage::kColorAttachmentOutput;
                    sync.mSrcAccess = rhi::Access::kColorAttachmentWrite;
                    sync.mDstStage = rhi::PipelineStage::kFragmentShader;
                    sync.mDstAccess = rhi::Access::kShaderRead;
                    cmd.ImageBarrier(*target->mImage, rhi::ImageLayout::kColorAttachment,
                            rhi::ImageLayout::kShaderReadOnly, sync);
                    target->mResolveLayout = rhi::ImageLayout::kShaderReadOnly;
                }
            } else if (target->mColorLayout != rhi::ImageLayout::kShaderReadOnly) {
                sync.mSrcStage = rhi::PipelineStage::kColorAttachmentOutput;
                sync.mSrcAccess = rhi::Access::kColorAttachmentWrite;
                sync.mDstStage = rhi::PipelineStage::kFragmentShader;
                sync.mDstAccess = rhi::Access::kShaderRead;
                cmd.ImageBarrier(*target->mImage, rhi::ImageLayout::kColorAttachment,
                        rhi::ImageLayout::kShaderReadOnly, sync);
                target->mColorLayout = rhi::ImageLayout::kShaderReadOnly;
            }
        }
        if (impl.mDepthTargetPtr != nullptr) {
            impl.ReleaseDepthAttachment(*impl.mDepthTargetPtr);
            impl.mDepthTargetPtr = nullptr;
        }
    }

    // ---- immediate-mode draw ----

    void Renderer::DrawImmediate(const UploadedMesh* mesh, const rhi::ShaderProgram& program,
            rhi::PrimitiveTopology topology, uint32_t instanceCount) {
        Impl& impl = *mImpl;
        rhi::CommandList& cmd = *impl.mCmd;

        if (!impl.mPassOpen) {
            moe::Error::Set("Draw: no active pass");
            return;
        }

        // pipeline: collect/cache; rebind when the hash changes or when a
        // push constant of this program changed since the last draw (the
        // rebind is what records the values into the command buffer)
        const bool useInstancing = instanceCount > 1;
        const rhi::GraphicsPipelineState state =
                impl.BuildPipelineState(mesh, program, topology, useInstancing);
        const uint64_t hash = state.GetHash();
        if (hash != impl.mCurrentPipelineHash || impl.HasDirtyFields(program)) {
            rhi::GraphicsPipeline pipeline;
            if (!impl.mDevice->GetOrCreateGraphicsPipeline(state, pipeline)) {
                moe::Error::Set("Draw: pipeline: " + moe::Error::Get());
                return;
            }
            cmd.BindGraphicsPipeline(pipeline);

            // replay push constants: per field, only those this program
            // declares and the user has set (fields of different programs may
            // share offsets; each keeps its own value)
            for (auto& field : impl.mFields) {
                if (field.mProgram != &program || field.mValue.empty()) {
                    continue;
                }
                cmd.SetPushConstants(pipeline, field.mOffset, field.mSize, field.mValue.data());
                field.mDirty = false;
            }

            impl.mCurrentPipelineHash = hash;
        }

        // images/samplers: rebuild the descriptor set per draw (correct and
        // cheap at teaching scale)
        if (impl.mImageCount > 0 || impl.mSamplerCount > 0 || impl.mBufferCount > 0) {
            rhi::GraphicsPipeline pipeline;
            if (impl.mDevice->GetOrCreateGraphicsPipeline(state, pipeline)) {
                rhi::DescriptorSetLayout layout;
                if (pipeline.GetDescriptorSetLayout(0, layout)) {
                    auto set = std::make_unique<rhi::DescriptorSet>();
                    if (impl.mDevice->CreateDescriptorSet(layout, *set)) {
                        bool written = true;
                        for (uint32_t i = 0; i < impl.mImageCount; ++i) {
                            written &= set->WriteImage(impl.mImages[i].mBinding,
                                    *impl.mImages[i].mImage, rhi::DescriptorType::kSampledImage);
                        }
                        for (uint32_t i = 0; i < impl.mSamplerCount; ++i) {
                            written &= set->WriteSampler(impl.mSamplers[i].mBinding,
                                    *impl.mSamplers[i].mSampler);
                        }
                        for (uint32_t i = 0; i < impl.mBufferCount; ++i) {
                            written &= set->WriteBuffer(impl.mBuffers[i].mBinding,
                                    *impl.mBuffers[i].mBuffer);
                        }
                        if (written) {
                            cmd.BindDescriptorSet(pipeline, *set, 0);
                            impl.mFrameSets.push_back(std::move(set));
                        } else {
                            moe::Error::Set("Draw: descriptor write failed "
                                    "(binding not declared in the shader?)");
                            set->Destroy();
                        }
                    }
                }
            }
        }

        if (mesh != nullptr) {
            cmd.BindVertexBuffer(mesh->mVertexBuffer, 0);
            if (impl.mInstance.mBuffer != nullptr) {
                cmd.BindVertexBuffer(*impl.mInstance.mBuffer, 1);
            }
            cmd.BindIndexBuffer(mesh->mIndexBuffer);
            cmd.DrawIndexed(mesh->mIndexCount, instanceCount, 0, 0, 0);
        } else {
            cmd.Draw(3, 1, 0, 0);
        }
        impl.mStatsDraws += 1;
    }

    // ---- PassContext: forwards to the renderer's internals ----

    void PassContext::SetState(const DrawState& state) {
        mRenderer->SetStateInternal(state);
    }

    void PassContext::SetCamera(const Camera& camera) {
        mRenderer->SetCameraInternal(camera);
    }

    void PassContext::ClearTextureBindings() {
        mRenderer->ClearTextureBindingsInternal();
    }

    int32_t PassContext::GetPushConstant(const rhi::ShaderProgram& program, const char* name) const {
        return mRenderer->GetPushConstant(program, name);
    }

    uint32_t PassContext::GetPushConstantSize(int32_t index) const {
        return mRenderer->GetPushConstantSize(index);
    }

    void PassContext::SetPushConstant(int32_t index, const void* data, size_t size) {
        mRenderer->SetPushConstantInternal(index, data, size);
    }

    void PassContext::BindImage(uint32_t binding, const rhi::Image& image) {
        mRenderer->BindImageInternal(binding, image);
    }

    void PassContext::BindSampler(uint32_t binding, const rhi::Sampler& sampler) {
        mRenderer->BindSamplerInternal(binding, sampler);
    }

    void PassContext::BindBuffer(uint32_t binding, const rhi::Buffer& buffer) {
        mRenderer->BindBufferInternal(binding, buffer);
    }

    void PassContext::BindInstanceBuffer(const rhi::Buffer& buffer, uint32_t stride,
            const InstanceAttribute* attributes, uint32_t attributeCount) {
        mRenderer->BindInstanceBufferInternal(buffer, stride, attributes, attributeCount);
    }

    void PassContext::Draw(const UploadedMesh& mesh, const rhi::ShaderProgram& program,
            rhi::PrimitiveTopology topology, uint32_t instanceCount) {
        mRenderer->DrawImmediate(&mesh, program, topology, instanceCount);
    }

    void PassContext::DrawFullscreen(const rhi::ShaderProgram& program) {
        mRenderer->DrawImmediate(nullptr, program, rhi::PrimitiveTopology::kTriangleList, 1);
    }
}// namespace moe::neo
