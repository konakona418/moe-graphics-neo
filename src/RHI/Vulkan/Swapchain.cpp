#include "RHI/Swapchain.hpp"

#include "RHI/CommandList.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <algorithm>

namespace moe::rhi {
    Swapchain::Swapchain() = default;

    Swapchain::~Swapchain() {
        MOE_RHI_ASSERT(mImpl == nullptr, "Swapchain leaked: Destroy() not called");
    }

    void Swapchain::Destroy() {
        if (mImpl && mImpl->mDevice && mImpl->mSwapchain != VK_NULL_HANDLE) {
            vkDestroyFence(mImpl->mDevice->mDevice, mImpl->mInFlight, nullptr);
            for (const auto semaphore : mImpl->mRenderFinished) {
                vkDestroySemaphore(mImpl->mDevice->mDevice, semaphore, nullptr);
            }
            vkDestroySemaphore(mImpl->mDevice->mDevice, mImpl->mImageAvailable, nullptr);
            for (const auto view : mImpl->mImageViews) {
                vkDestroyImageView(mImpl->mDevice->mDevice, view, nullptr);
            }
            vkDestroySwapchainKHR(mImpl->mDevice->mDevice, mImpl->mSwapchain, nullptr);
        }
        mImpl.reset();
    }

    bool Swapchain::AcquireImage() {
        if (mImpl == nullptr || mImpl->mSwapchain == VK_NULL_HANDLE) {
            return false;
        }
        vkWaitForFences(mImpl->mDevice->mDevice, 1, &mImpl->mInFlight, VK_TRUE, UINT64_MAX);
        vkResetFences(mImpl->mDevice->mDevice, 1, &mImpl->mInFlight);

        const VkResult result = vkAcquireNextImageKHR(mImpl->mDevice->mDevice,
                mImpl->mSwapchain, UINT64_MAX, mImpl->mImageAvailable, VK_NULL_HANDLE,
                &mImpl->mCurrentImage);
        return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
    }

    bool Swapchain::BeginRendering(CommandList& cmd, const float clearColor[4]) {
        if (mImpl == nullptr || mImpl->mCurrentImage >= mImpl->mImages.size()) {
            return false;
        }

        // Acquired image (treated as Undefined, contents discarded) -> color attachment.
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = mImpl->mImages[mImpl->mCurrentImage];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd.mImpl->mCommandBuffer, &dependency);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = mImpl->mImageViews[mImpl->mCurrentImage];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {
                {clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = {{0, 0}, {mImpl->mWidth, mImpl->mHeight}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        vkCmdBeginRendering(cmd.mImpl->mCommandBuffer, &renderingInfo);
        return true;
    }

    void Swapchain::EndRendering(CommandList& cmd) {
        vkCmdEndRendering(cmd.mImpl->mCommandBuffer);

        // color attachment -> presentable
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_NONE;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = mImpl->mImages[mImpl->mCurrentImage];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd.mImpl->mCommandBuffer, &dependency);
    }

    bool Swapchain::Present(CommandList& cmd) {
        if (mImpl == nullptr || mImpl->mSwapchain == VK_NULL_HANDLE
                || mImpl->mCurrentImage >= mImpl->mRenderFinished.size()) {
            return false;
        }

        VkSemaphore renderFinished = mImpl->mRenderFinished[mImpl->mCurrentImage];

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &mImpl->mImageAvailable;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd.mImpl->mCommandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished;
        if (vkQueueSubmit(mImpl->mDevice->mGraphicsQueue, 1, &submit, mImpl->mInFlight) != VK_SUCCESS) {
            return false;
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderFinished;
        present.swapchainCount = 1;
        present.pSwapchains = &mImpl->mSwapchain;
        present.pImageIndices = &mImpl->mCurrentImage;
        return vkQueuePresentKHR(mImpl->mDevice->mGraphicsQueue, &present) == VK_SUCCESS;
    }

    uint32_t Swapchain::GetWidth() const {
        return mImpl ? mImpl->mWidth : 0;
    }

    uint32_t Swapchain::GetHeight() const {
        return mImpl ? mImpl->mHeight : 0;
    }

    Format Swapchain::GetFormat() const {
        return mImpl ? mImpl->mFormat : Format::kUndefined;
    }

    bool Swapchain::GetCurrentImage(Image& outImage) {
        if (mImpl == nullptr || mImpl->mCurrentImage >= mImpl->mImages.size()
                || mImpl->mCurrentImage >= mImpl->mImageViews.size()) {
            return false;
        }
        outImage.mImpl = std::make_unique<ImageImpl>();
        auto* impl = outImage.mImpl.get();
        impl->mDevice = mImpl->mDevice;
        impl->mImage = mImpl->mImages[mImpl->mCurrentImage];
        impl->mView = mImpl->mImageViews[mImpl->mCurrentImage];
        impl->mType = ImageType::k2D;
        impl->mWidth = mImpl->mWidth;
        impl->mHeight = mImpl->mHeight;
        impl->mDepth = 1;
        impl->mMipLevels = 1;
        impl->mLayerCount = 1;
        impl->mFormat = mImpl->mFormat;
        impl->mOwned = false;
        return true;
    }
}// namespace moe::rhi