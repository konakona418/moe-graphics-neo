#include "Neo/SwapchainImage.hpp"

namespace moe::neo {
    SwapchainImage::~SwapchainImage() {
        Release();
    }

    bool SwapchainImage::Acquire(rhi::Swapchain& swapchain) {
        Release();
        if (!swapchain.GetCurrentImage(mImage)) {
            return false;
        }
        mSwapchain = &swapchain;
        mFormat = swapchain.GetFormat();
        mWidth = swapchain.GetWidth();
        mHeight = swapchain.GetHeight();
        return true;
    }

    void SwapchainImage::Release() {
        if (mSwapchain != nullptr) {
            mImage.Destroy(); // borrowed wrapper: only drops the wrapper
            mSwapchain = nullptr;
            mFormat = rhi::Format::kUndefined;
            mWidth = 0;
            mHeight = 0;
        }
    }

    bool SwapchainImage::IsValid() const {
        return mSwapchain != nullptr;
    }

    const rhi::Image& SwapchainImage::GetImage() const {
        return mImage;
    }

    rhi::Format SwapchainImage::GetFormat() const {
        return mFormat;
    }

    uint32_t SwapchainImage::GetWidth() const {
        return mWidth;
    }

    uint32_t SwapchainImage::GetHeight() const {
        return mHeight;
    }
}// namespace moe::neo
