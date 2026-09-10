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

#include "Neo/Assets.hpp"
#include "Neo/Cache.hpp"
#include "Neo/SwapchainImage.hpp"
#include "Neo/Uploader.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
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
    // the attachment/sampling cycle are handled internally. With MSAA the
    // pass renders into mMsaaImage and resolves into the sampleable mImage.
    struct RenderTarget {
        std::unique_ptr<rhi::Image> mImage;      // single-sample, sampleable (resolve target when MS)
        std::unique_ptr<rhi::Image> mMsaaImage;  // valid when mSampleCount > 1 (render attachment)
        std::unique_ptr<rhi::Image> mDepthImage; // valid when mHasDepth
        rhi::Format mFormat{rhi::Format::kR8G8B8A8Unorm};
        uint32_t mWidth{0};
        uint32_t mHeight{0};
        uint32_t mSampleCount{1};
        bool mHasDepth{false};

        // Internal barrier bookkeeping (read-only for users).
        rhi::ImageLayout mColorLayout{rhi::ImageLayout::kUndefined};
        rhi::ImageLayout mResolveLayout{rhi::ImageLayout::kUndefined};
        rhi::ImageLayout mDepthLayout{rhi::ImageLayout::kUndefined};

        // The image a pass renders into (multisampled when MSAA is on).
        rhi::Image& AttachmentImage() { return mMsaaImage ? *mMsaaImage : *mImage; }
    };

    using RenderTargetHandle = Handle<RenderTarget>;

    // Which image of a render target a pass attachment uses.
    enum class AttachmentKind { kColor, kDepth };

    // A pass attachment: one image of a render target plus its load op. An
    // invalid target in a color attachment means the swapchain image; an
    // invalid target in a depth attachment means "the color attachment's own
    // depth" (or the renderer's main depth for swapchain passes). Any
    // attachment may come from any target (borrowed): the renderer transitions
    // the owner's tracked layout before the pass and synchronizes the previous
    // writer, then returns the image to ShaderReadOnly after the pass so later
    // passes can sample it.
    struct PassAttachment {
        RenderTargetHandle mTarget;
        AttachmentKind mKind{AttachmentKind::kColor};
        rhi::LoadOp mLoadOp{rhi::LoadOp::kClear};
    };

    // Convenience builders for the common cases.
    inline PassAttachment ColorAttachment(RenderTargetHandle target,
            rhi::LoadOp loadOp = rhi::LoadOp::kClear) {
        return {target, AttachmentKind::kColor, loadOp};
    }

    inline PassAttachment DepthAttachment(RenderTargetHandle target,
            rhi::LoadOp loadOp = rhi::LoadOp::kLoad) {
        return {target, AttachmentKind::kDepth, loadOp};
    }

    // Pass description: a render pass is data. Built once in setup, executed
    // every frame via renderer.Execute. Both attachments default to the
    // swapchain (color) and its main depth.
    struct PassDesc {
        const char* mName{nullptr};
        PassAttachment mColor;
        PassAttachment mDepth;
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
        void BindBuffer(uint32_t binding, const rhi::Buffer& buffer);
        void BindInstanceBuffer(const rhi::Buffer& buffer, uint32_t stride,
                const InstanceAttribute* attributes, uint32_t attributeCount);
        void Draw(const UploadedMesh& mesh, const rhi::ShaderProgram& program,
                rhi::PrimitiveTopology topology = rhi::PrimitiveTopology::kTriangleList,
                uint32_t instanceCount = 1);
        // Raw vertex draw for callers with their own vertex layout (e.g.
        // text): the buffer is bound at binding 0 and `attributes` are the
        // shader's declared locations/formats/offsets, exactly as reflected
        // from SPIR-V. No index buffer is used. `firstVertex` offsets into
        // the buffer (draws may share one frame arena).
        void DrawVertices(const rhi::Buffer& vertexBuffer, uint32_t vertexCount,
                const rhi::VertexAttribute* attributes, uint32_t attributeCount,
                uint32_t stride, const rhi::ShaderProgram& program,
                rhi::PrimitiveTopology topology = rhi::PrimitiveTopology::kTriangleList,
                uint32_t firstVertex = 0);
        void DrawFullscreen(const rhi::ShaderProgram& program);

        // ---- content layer ----

        // Camera for DrawModel (sticky for the frame). Feeds viewProj/view/
        // cameraPos to programs that declare those push constant names.
        void SetCamera(const Camera& camera);

        // Drops every image/sampler/buffer binding (DrawModel calls this per
        // primitive so stale material textures never leak into the next one).
        void ClearTextureBindings();

        // Push constant name lookup/size, forwarded from the renderer (used
        // by DrawModel and by demos that set their own named fields).
        int32_t GetPushConstant(const rhi::ShaderProgram& program, const char* name) const;
        uint32_t GetPushConstantSize(int32_t index) const;

        // Draws every part of a model: node transforms are applied, each
        // primitive uses its material's program (falling back to
        // defaultProgram), and material textures/parameters are fed by name.
        void DrawModel(const Model& model, ProgramHandle defaultProgram,
                const glm::mat4& transform);
        // Same, but every part uses `program` (e.g. the outline pass).
        void DrawModelForced(const Model& model, ProgramHandle program,
                const glm::mat4& transform);
        // Draws only the parts whose material name matches (escape hatch).
        void DrawModelPart(const Model& model, const char* materialName, ProgramHandle program,
                const glm::mat4& transform);

        // Text draw from glyph outlines: lays out UTF-8 `text` (word order
        // left to right, '\n' breaks lines), builds the glyph vertices and
        // draws them with `program` (the text shader). The program must
        // declare the push constant names viewProj/model/viewport and the
        // storage buffers `curveData`/`bandData`. The shader outputs
        // premultiplied coverage, so draw with premultiplied blending
        // (src = One, dst = OneMinusSrcAlpha).
        void DrawText(const Font& font, std::string_view text, const TextDrawParams& params,
                ProgramHandle program);

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
    //   PassDesc scene{"scene", {}, {}};
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
                uint32_t width, uint32_t height, uint32_t sampleCount = 1);
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
        // the desc+continuation core (color attachment on mTarget, default
        // depth).
        template<typename T>
        void ExecutePass(T& pass) {
            PassDesc desc{};
            desc.mName = T::kName;
            desc.mColor = ColorAttachment(pass.mTarget, pass.mLoadOp);
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

        // Size in bytes of a field resolved by GetPushConstant (0 = invalid).
        uint32_t GetPushConstantSize(int32_t index) const;

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

        // Render targets default to the renderer's MSAA level; pass sampleCount
        // (1/2/4/8) to override it for this target (e.g. a 1x normal prepass
        // whose edges must stay crisp). 0 = renderer default.
        RenderTargetHandle CreateRenderTarget(uint32_t width, uint32_t height,
                rhi::Format format, bool withDepth, uint32_t sampleCount = 0);
        void DestroyRenderTarget(RenderTargetHandle handle);
        RenderTarget* GetRenderTarget(RenderTargetHandle handle);

        // MSAA level of this renderer (1 = off).
        uint32_t GetSampleCount() const;

    private:
        friend class PassContext;
        void SetStateInternal(const DrawState& state);
        void SetCameraInternal(const Camera& camera);
        const Camera* GetCameraInternal() const;
        void ClearTextureBindingsInternal();
        void BindImageInternal(uint32_t binding, const rhi::Image& image);
        void BindSamplerInternal(uint32_t binding, const rhi::Sampler& sampler);
        void BindBufferInternal(uint32_t binding, const rhi::Buffer& buffer);
        void BindInstanceBufferInternal(const rhi::Buffer& buffer, uint32_t stride,
                const InstanceAttribute* attributes, uint32_t attributeCount);
        void BeginPass(const PassDesc& desc);
        void EndPass(const PassDesc& desc);
        // immediate-mode draw entry (called from PassContext)
        void DrawImmediate(const UploadedMesh* mesh, const rhi::ShaderProgram& program,
                rhi::PrimitiveTopology topology, uint32_t instanceCount);
        void DrawVerticesImmediate(const rhi::Buffer& vertexBuffer, uint32_t vertexCount,
                const rhi::VertexAttribute* attributes, uint32_t attributeCount,
                uint32_t stride, const rhi::ShaderProgram& program,
                rhi::PrimitiveTopology topology, uint32_t firstVertex);
        // Size in pixels of the pass currently being recorded.
        glm::vec2 GetViewportSizeInternal() const;
        // Appends per-frame dynamic vertices (text) to the frame arena and
        // returns the byte offset of the appended block, or UINT32_MAX on
        // failure. The arena is uploaded once in EndFrame, before the frame
        // is submitted, so recorded draws read stable data.
        uint32_t AppendDynamicVertices(const void* data, uint32_t bytes);
        const rhi::Buffer& GetDynamicVertexBufferInternal() const;
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
