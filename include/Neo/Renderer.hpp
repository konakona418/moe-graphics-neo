#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/PipelineState.hpp>
#include <RHI/Sampler.hpp>
#include <RHI/Shader.hpp>

#include "Neo/Cache.hpp"
#include "Neo/SwapchainImage.hpp"
#include "Neo/Uploader.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace moe::neo {
    constexpr uint32_t kMaxTextureBindings = 8;

    // Rendering intent (GL-style state). State is set inside a pass context,
    // sticky across passes (unset values carry over), and recorded commands
    // are immediately final — changing state never retroactively affects
    // already-drawn draws (the classic GL "forgot to change state" bug is
    // structurally impossible here).
    struct DrawState {
        bool mDepthTest{true};
        bool mDepthWrite{true};
        rhi::CompareOp mDepthCompareOp{rhi::CompareOp::kLess};
        rhi::CullMode mCullMode{rhi::CullMode::kBack};
        rhi::FrontFace mFrontFace{rhi::FrontFace::kCounterClockwise};
        rhi::PolygonMode mPolygonMode{rhi::PolygonMode::kFill};
        bool mBlendEnabled{false};
        rhi::BlendFactor mBlendSrcColor{rhi::BlendFactor::kSrcAlpha};
        rhi::BlendFactor mBlendDstColor{rhi::BlendFactor::kOneMinusSrcAlpha};
        rhi::BlendOp mBlendColorOp{rhi::BlendOp::kAdd};
        rhi::BlendFactor mBlendSrcAlpha{rhi::BlendFactor::kOne};
        rhi::BlendFactor mBlendDstAlpha{rhi::BlendFactor::kOneMinusSrcAlpha};
        rhi::BlendOp mBlendAlphaOp{rhi::BlendOp::kAdd};
    };

    // Per-instance vertex input declaration for BindInstanceBuffer. The
    // layout is fully explicit: location/format/offset mirror what the shader
    // declares; there are no implicit conventions.
    struct InstanceAttribute {
        uint32_t mLocation{0};
        rhi::Format mFormat{rhi::Format::kR32G32B32A32Float};
        uint32_t mOffset{0};
    };

    // Render target. Created/destroyed through the renderer; lifetime is
    // managed by a Cache keyed by RenderTargetHandle (stale handles resolve
    // to null). The target's color/depth images can be sampled in later
    // passes (pass mImage/mDepthImage to BindImage). Layout transitions for
    // the attachment/sampling cycle are handled internally.
    struct RenderTarget {
        std::unique_ptr<rhi::Image> mImage;
        std::unique_ptr<rhi::Image> mDepthImage; // valid when mHasDepth
        rhi::Format mFormat{rhi::Format::kR8G8B8A8Unorm};
        uint32_t mWidth{0};
        uint32_t mHeight{0};
        bool mHasDepth{false};

        // Internal barrier bookkeeping (read-only for users).
        rhi::ImageLayout mColorLayout{rhi::ImageLayout::kUndefined};
        rhi::ImageLayout mDepthLayout{rhi::ImageLayout::kUndefined};
    };

    using RenderTargetHandle = Handle<RenderTarget>;

    // Pass description: a render pass is data. Built once in setup, executed
    // every frame via renderer.Execute. An invalid mTarget means the
    // swapchain.
    struct PassDesc {
        const char* mName{nullptr};
        RenderTargetHandle mTarget;
        rhi::LoadOp mLoadOp{rhi::LoadOp::kClear};
    };

    class Renderer;

    // Pass-scope rendering context: every state/draw call lives here, so a
    // draw outside a pass is a compile-time error. Calls operate on the
    // renderer's per-frame state (sticky across passes).
    class PassContext {
    public:
        void SetState(const DrawState& state);
        void SetPushConstant(int32_t index, const void* data, size_t size);
        void BindImage(uint32_t binding, const rhi::Image& image);
        void BindSampler(uint32_t binding, const rhi::Sampler& sampler);
        void BindInstanceBuffer(const rhi::Buffer& buffer, uint32_t stride,
                const InstanceAttribute* attributes, uint32_t attributeCount);
        void Draw(const UploadedMesh& mesh, const rhi::ShaderProgram& program,
                rhi::PrimitiveTopology topology = rhi::PrimitiveTopology::kTriangleList,
                uint32_t instanceCount = 1);
        void DrawFullscreen(const rhi::ShaderProgram& program);

    private:
        friend class Renderer;
        explicit PassContext(Renderer& renderer) : mRenderer(&renderer) {}
        Renderer* mRenderer{nullptr};
    };

    // Convenience layer over the RHI ("GL layer", immediate mode): explicit
    // passes (PassDesc + Execute), GL-style state calls, automatic pipeline
    // collection/caching, name-addressed push constants, off-screen targets
    // and manual per-draw texture binding. Commands are recorded immediately
    // as draws are issued; user code can freely interleave raw RHI commands
    // between passes (e.g. compute dispatches). Engine-managed barriers are
    // exposed as thin proxies (ImageBarrier/BufferBarrier/MemoryBarrier) so
    // the internal layout bookkeeping stays in sync.
    //
    // Usage (inside App's mPostRender callback; no render pass active):
    //   PassDesc scene{"scene", {}, rhi::LoadOp::kClear};
    //   renderer.BeginFrame(cmd, frame, clearColor);
    //   renderer.Execute(scene, [&](PassContext& pass) {
    //       pass.SetPushConstant(mvpIdx, &mvp);
    //       pass.Draw(mesh, program);
    //   });
    //   renderer.EndFrame(); // ends any leftover pass
    class Renderer {
    public:
        Renderer();
        ~Renderer();

        Renderer(const Renderer&) = delete;
        Renderer& operator=(const Renderer&) = delete;

        bool Init(rhi::Device& device, rhi::DefaultPipelineCache& cache,
                uint32_t width, uint32_t height, std::string& error);
        void Destroy();

        // Begins a frame; `frame` is the current swapchain image (acquired
        // via SwapchainImage::Acquire, valid until Release after EndFrame).
        // The swapchain image's layout is tracked by the Swapchain itself, so
        // any starting layout (Undefined after acquire, PresentSrc after a
        // previous frame) is handled. Per-frame state is reset here.
        void BeginFrame(rhi::CommandList& cmd, const SwapchainImage& frame,
                const float clearColor[4]);

        // Executes one pass: transitions the target, begins rendering (with
        // the pass's load op), runs the continuation against a PassContext,
        // then ends the pass. Passes never nest (asserted).
        template<typename F>
        void Execute(const PassDesc& desc, F&& body) {
            BeginPass(desc);
            PassContext context(*this);
            std::forward<F>(body)(context);
            EndPass(desc);
        }

        // CRTP pass convenience: passes deriving from Pass<T> declare kName,
        // mTarget, mLoadOp and Execute(PassContext&); this converts them to
        // the desc+continuation core.
        template<typename T>
        void ExecutePass(T& pass) {
            PassDesc desc{};
            desc.mName = T::kName;
            desc.mTarget = pass.mTarget;
            desc.mLoadOp = pass.mLoadOp;
            Execute(desc, [&](PassContext& context) { pass.Execute(context); });
        }

        // Closes any leftover pass (swapchain passes end through the
        // Swapchain, which returns the image to PresentSrc) and logs
        // per-second stats.
        void EndFrame();

        // Resolves a push constant member name of `program` to an index for
        // this renderer (-1 when absent). Indexes are cached; look up once in
        // setup, then SetPushConstant every frame.
        int32_t GetPushConstant(const rhi::ShaderProgram& program, const char* name) const;

        // Engine-recognized barriers: thin proxies over the RHI that keep the
        // internal layout bookkeeping in sync (an ImageBarrier on a render
        // target's image updates its tracked layout). Raw RHI barriers on
        // render targets bypass the bookkeeping and are the user's own
        // responsibility.
        void ImageBarrier(const rhi::Image& image, rhi::ImageLayout srcLayout,
                rhi::ImageLayout dstLayout, const rhi::SyncInfo& sync);
        void BufferBarrier(const rhi::Buffer& buffer, const rhi::SyncInfo& sync);
        void MemoryBarrier(const rhi::SyncInfo& sync);

        // ---- render targets ----

        RenderTargetHandle CreateRenderTarget(uint32_t width, uint32_t height,
                rhi::Format format, bool withDepth, std::string& error);
        void DestroyRenderTarget(RenderTargetHandle handle);
        RenderTarget* GetRenderTarget(RenderTargetHandle handle);

        const std::string& GetLastError() const;

    private:
        friend class PassContext;
        void SetStateInternal(const DrawState& state);
        void BindImageInternal(uint32_t binding, const rhi::Image& image);
        void BindSamplerInternal(uint32_t binding, const rhi::Sampler& sampler);
        void BindInstanceBufferInternal(const rhi::Buffer& buffer, uint32_t stride,
                const InstanceAttribute* attributes, uint32_t attributeCount);
        void BeginPass(const PassDesc& desc);
        void EndPass(const PassDesc& desc);
        // immediate-mode draw entry (called from PassContext)
        void DrawImmediate(const UploadedMesh* mesh, const rhi::ShaderProgram& program,
                rhi::PrimitiveTopology topology, uint32_t instanceCount);
        bool SetPushConstantInternal(int32_t index, const void* data, size_t size);
        int32_t LookupField(const rhi::ShaderProgram& program, const char* name);

        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };

    // CRTP base for structured passes: derive, declare
    //   static constexpr const char* kName;
    //   RenderTargetHandle mTarget;
    //   rhi::LoadOp mLoadOp;
    // and implement void Execute(PassContext&). Run through
    // renderer.ExecutePass(pass).
    template<typename T>
    struct Pass {
        void Run(Renderer& renderer) {
            renderer.ExecutePass(*static_cast<T*>(this));
        }
    };
}// namespace moe::neo
