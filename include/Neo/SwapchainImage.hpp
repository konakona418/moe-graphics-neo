#pragma once

#include <RHI/Image.hpp>
#include <RHI/Swapchain.hpp>

namespace moe::neo {
    // The presentable image of the current swapchain frame: a borrowed
    // wrapper that lives for exactly one frame (Acquire ... Release). The
    // image itself is owned by the swapchain; this object only carries the
    // handle + metadata so the "current frame image" is an explicit object
    // instead of a raw RHI::Image passed around.
    class SwapchainImage {
    public:
        SwapchainImage() = default;
        ~SwapchainImage();

        SwapchainImage(const SwapchainImage&) = delete;
        SwapchainImage& operator=(const SwapchainImage&) = delete;

        // Binds the swapchain's current image (call right after
        // swapchain.AcquireImage()). The previous binding is dropped.
        bool Acquire(rhi::Swapchain& swapchain);

        // Drops the borrowed wrapper; the object returns to invalid.
        void Release();

        bool IsValid() const;
        const rhi::Image& GetImage() const;
        rhi::Format GetFormat() const;
        uint32_t GetWidth() const;
        uint32_t GetHeight() const;

    private:
        rhi::Image mImage;
        rhi::Swapchain* mSwapchain{nullptr};
        rhi::Format mFormat{rhi::Format::kUndefined};
        uint32_t mWidth{0};
        uint32_t mHeight{0};
    };
}// namespace moe::neo
