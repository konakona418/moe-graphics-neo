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
#include "Neo/Uploader.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace moe::neo {
    constexpr uint32_t kMaxTextureBindings = 8;

    // Rendering intent (GL-style state). Set with SetState: sticky per frame,
    // but every Draw snapshots the state it sees, so later changes never
    // affect already-queued draws (the classic GL "forgot to change state"
    // bug is structurally impossible here).
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

    // Render target. Created/destroyed through the renderer;
    // lifetime is managed by a Cache keyed by RenderTargetHandle (stale handles
    // resolve to null). The target's color image can be sampled in later
    // draws (pass mImage.get() to BindImage). Layout transitions for the
    // attachment/sampling cycle are handled internally.
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

    // Convenience layer over the RHI ("GL layer"): GL-style state calls,
    // automatic pipeline collection/caching, implicit batching (draws are
    // queued and recorded sorted at EndFrame), name-addressed push constants
    // and render targets. No magic: every piece of data flowing into a
    // shader is written by an explicit call; names/bindings are chosen by the
    // user (shader and C++ side by side, like GL).
    //
    // Usage (inside App's mPostRender callback; no render pass active):
    //   renderer.BeginFrame(cmd, swapImage, swapchainFormat, clearColor);
    //   renderer.SetState(state);
    //   renderer.SetPushConstant(mvpIdx, &mvp);
    //   renderer.BindImage(0, &textureImage);
    //   renderer.BindSampler(1, &textureSampler);
    //   renderer.Draw(mesh, program);
    //   renderer.EndFrame(); // sorts, records, transitions back to PresentSrc
    class Renderer {
    public:
        Renderer();
        ~Renderer();

        Renderer(const Renderer&) = delete;
        Renderer& operator=(const Renderer&) = delete;

        bool Init(rhi::Device& device, rhi::DefaultPipelineCache& cache,
                uint32_t width, uint32_t height, std::string& error);
        void Destroy();

        // Begins a frame; the swapchain image must currently be in PresentSrc
        // layout (as App leaves it after EndRendering). Queues and the
        // per-frame state are reset here.
        void BeginFrame(rhi::CommandList& cmd, const rhi::Image& swapchainImage,
                rhi::Format swapchainFormat, const float clearColor[4]);

        // Sorts queued draws (render-target/pipeline-switch minimization),
        // records them into the command list handed to BeginFrame, and
        // transitions the swapchain image back to PresentSrc.
        void EndFrame();

        // ---- GL-style state calls (sticky; each Draw snapshots them) ----

        void SetState(const DrawState& state);

        // Selects the render target for subsequent draws ({} = swapchain).
        void BindTarget(RenderTargetHandle target);

        // Binds a sampled image / sampler to a descriptor binding. The
        // binding numbers must match what the shader declares (e.g.
        // [vk::binding(0, 0)] Texture2D ... / [vk::binding(1, 0)] SamplerState
        // ...). Images that belong to an render target get their layout
        // transitioned automatically when drawn to / sampled.
        void BindImage(uint32_t binding, const rhi::Image& image);
        void BindSampler(uint32_t binding, const rhi::Sampler& sampler);

        // Binds a per-instance vertex buffer (binding 1): instanceCount
        // records of stride bytes; attributes mirror the shader's instance
        // inputs. Cleared on BeginFrame; re-bind before instanced draws.
        void BindInstanceBuffer(const rhi::Buffer& buffer, uint32_t stride,
                const InstanceAttribute* attributes, uint32_t attributeCount);

        // Writes data into the per-frame push constant value table; the next
        // Draw() snapshots it. size must equal the reflected member size.
        bool SetPushConstant(int32_t index, const void* data, size_t size);

        // Resolves a push constant member name of `program` to an index for
        // this renderer (-1 when absent). Indexes are cached; look up once in
        // setup, then SetPushConstant every frame.
        int32_t GetPushConstant(const rhi::ShaderProgram& program, const char* name) const;

        // ---- draws ----

        // Queues one draw using the current frame state. The pipeline
        // (program + vertex layout + state + target format) is collected
        // lazily and cached. `mesh` must outlive the frame.
        void Draw(const UploadedMesh& mesh, const rhi::ShaderProgram& program,
                rhi::PrimitiveTopology topology = rhi::PrimitiveTopology::kTriangleList,
                uint32_t instanceCount = 1);

        // Full-screen triangle (no vertex buffer; the shader generates
        // SV_VertexID). Meant for post-processing passes.
        void DrawFullscreen(const rhi::ShaderProgram& program);

        // ---- render targets ----

        RenderTargetHandle CreateRenderTarget(uint32_t width, uint32_t height,
                rhi::Format format, bool withDepth, std::string& error);
        void DestroyRenderTarget(RenderTargetHandle handle);
        RenderTarget* GetRenderTarget(RenderTargetHandle handle);

        const std::string& GetLastError() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
