#pragma once

#include "RHI/RHICommon.hpp"

#include <memory>

namespace moe::rhi {
    class Device;
    class CommandList;
    class Image;
    struct SwapchainImpl;

    // Present-capable image source. The VkSurfaceKHR is created by the caller
    // (e.g. neo::Window via GLFW) and passed in as an opaque handle. Single
    // frame in flight: per frame call AcquireImage, record between
    // BeginRendering/EndRendering, then Present(cmd).
    class Swapchain {
    public:
        Swapchain();
        ~Swapchain();
        Swapchain(const Swapchain&) = delete;
        Swapchain& operator=(const Swapchain&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the swapchain
        // was created but not destroyed (leak trap).
        void Destroy();

        // Waits for the in-flight frame and acquires the next image.
        bool AcquireImage();

        // Begins dynamic rendering into the current image, transitioning it
        // from whatever layout it currently has (Undefined after acquire,
        // PresentSrc after a previous EndRendering) to ColorAttachment.
        // loadOp kClear clears to clearColor, kLoad keeps the contents.
        // depthImage (optional) must already be in DepthStencilAttachment
        // layout; depthLoadOp kClear clears it to depthClear, kLoad borrows
        // the existing contents (e.g. a later pass depth-testing against an
        // earlier pass). Record draws between BeginRendering and EndRendering.
        bool BeginRendering(CommandList& cmd, const float clearColor[4],
                LoadOp loadOp = LoadOp::kClear, const Image* depthImage = nullptr,
                float depthClear = 1.0f, LoadOp depthLoadOp = LoadOp::kClear);
        void EndRendering(CommandList& cmd);

        // Transfer path: transitions the current image to TransferDst (for a
        // CopyImage/BlitImage into it), then back to PresentSrc. Layout
        // bookkeeping stays with the swapchain either way.
        bool BeginTransfer(CommandList& cmd);
        void EndTransfer(CommandList& cmd);

        // Submits the recorded command list (waiting on image availability,
        // signaling render completion) and presents the current image.
        bool Present(CommandList& cmd);

        uint32_t GetWidth() const;
        uint32_t GetHeight() const;
        Format GetFormat() const;

        // Wraps the currently acquired image as a non-owning Image (Destroy()
        // on the wrapper drops only the wrapper, never the swapchain image).
        // Usable as a CopyImage/BlitImage destination or ImageBarrier target.
        // Call after AcquireImage, before Present; the wrapper is invalidated
        // by the next AcquireImage.
        bool GetCurrentImage(Image& outImage);

    private:
        friend class Device;
        std::unique_ptr<SwapchainImpl> mImpl;
    };
}// namespace moe::rhi