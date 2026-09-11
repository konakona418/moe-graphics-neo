#pragma once

#include "RHI/RHICommon.hpp"

#include <cstdint>
#include <memory>

namespace moe::rhi {
    class Device;
    class Buffer;
    class Image;
    class GraphicsPipeline;
    class ComputePipeline;
    class DescriptorSet;
    struct CommandListImpl;

    class CommandList {
    public:
        CommandList();
        ~CommandList();

        CommandList(const CommandList&) = delete;
        CommandList& operator=(const CommandList&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the command
        // list was created but not destroyed (leak trap).
        void Destroy();

        void Begin();
        void End();
        // Copies size bytes from src at srcOffset to dst at dstOffset.
        void CopyBuffer(const Buffer& src, const Buffer& dst,
                uint64_t size, uint64_t srcOffset = 0, uint64_t dstOffset = 0);
        // Copies the whole image (currently in TransferSrc layout) to the
        // buffer. Image->buffer copy; buffer needs TransferDst usage.
        void CopyImageToBuffer(const Image& image, const Buffer& dst);
        // Uploads tightly-packed buffer data into the image subresource. The
        // image must currently be in TransferDst layout; buffer needs
        // TransferSrc usage.
        void CopyBufferToImage(const Buffer& src, const Image& dst,
                uint32_t mipLevel = 0, uint32_t baseArrayLayer = 0, uint32_t layerCount = 1,
                uint32_t bufferOffset = 0);
        // Copies the whole src image (in srcLayout) into dst (in dstLayout).
        // Same format required (use BlitImage for format conversion/scaling).
        void CopyImage(const Image& src, ImageLayout srcLayout,
                const Image& dst, ImageLayout dstLayout);
        // Scales src (in srcLayout) into dst (in dstLayout) with the given
        // filter. Allows format conversion and resizing.
        void BlitImage(const Image& src, ImageLayout srcLayout,
                const Image& dst, ImageLayout dstLayout, Filter filter);

        // Begins dynamic rendering to the given color image, optionally with a
        // depth attachment. The images must already be in their attachment
        // layouts (caller or the RenderGraph issues the transitions). Load =
        // clear with clearColor / depthClear, or load (keep contents). A
        // borrowed depth attachment (depthLoadOp = kLoad) is how a later pass
        // depth-tests against an earlier pass's depth. resolveColor (optional)
        // must be a single-sample image of the same size/format: a multisampled
        // `color` is resolved into it at the end of the pass.
        void BeginRendering(const Image& color, const float clearColor[4],
                const Image* depth = nullptr, float depthClear = 1.0f,
                LoadOp colorLoadOp = LoadOp::kClear, LoadOp depthLoadOp = LoadOp::kClear,
                const Image* resolveColor = nullptr);
        // Ends dynamic rendering; the image stays in ColorAttachment layout.
        void EndRendering();

        void BindDescriptorSet(const GraphicsPipeline& pipeline, const DescriptorSet& set, uint32_t setIndex);
        void BindDescriptorSet(const ComputePipeline& pipeline, const DescriptorSet& set, uint32_t setIndex);
        void SetPushConstants(const GraphicsPipeline& pipeline, uint32_t offset, size_t size, const void* data);
        void SetPushConstants(const ComputePipeline& pipeline, uint32_t offset, size_t size, const void* data);

        void BindGraphicsPipeline(const GraphicsPipeline& pipeline);
        void Dispatch(const ComputePipeline& pipeline, uint32_t x, uint32_t y, uint32_t z);

        // Sets the dynamic viewport + scissor to the full (width x height) area.
        void SetViewport(uint32_t width, uint32_t height);

        // Sets the dynamic scissor rectangle (framebuffer pixels, origin at the
        // top-left, y down). Dynamic state: no pipeline rebuild. A zero-size
        // rectangle clips everything. Resets to the full area on SetViewport.
        void SetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height);

        // Sets the dynamic stencil reference (front and back faces). Dynamic
        // state: no pipeline rebuild.
        void SetStencilReference(uint32_t reference);

        void BindVertexBuffer(const Buffer& buffer, uint32_t binding);
        void BindIndexBuffer(const Buffer& buffer);
        void Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
        void DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);

        // Records an explicit memory dependency: waits until the source stage's
        // accesses complete, then makes them visible to the destination stage.
        // MemoryBarrier is global (escape hatch); BufferBarrier and ImageBarrier
        // scope the dependency to one resource (ImageBarrier also migrates the
        // image layout).
        void MemoryBarrier(const SyncInfo& sync);
        void BufferBarrier(const Buffer& buffer, const SyncInfo& sync);
        void ImageBarrier(const Image& image, ImageLayout srcLayout, ImageLayout dstLayout, const SyncInfo& sync);

        // Opaque VkCommandBuffer handle for integrations that must talk to
        // Vulkan directly (e.g. ImGui). Same caveat as Device::GetVulkanHandles.
        uintptr_t GetVulkanHandle() const;

    private:
        friend class Device;
        friend class Swapchain;
        friend class BindlessSet;

        std::unique_ptr<CommandListImpl> mImpl;
    };
}// namespace moe::rhi