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

        // Begins dynamic rendering into the current image (cleared to
        // clearColor). Record draws between BeginRendering and EndRendering.
        bool BeginRendering(CommandList& cmd, const float clearColor[4]);
        void EndRendering(CommandList& cmd);

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