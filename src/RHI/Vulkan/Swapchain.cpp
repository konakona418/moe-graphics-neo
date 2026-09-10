#include "RHI/Swapchain.hpp"

#include "RHI/CommandList.hpp"
#include "Mappings.hpp"
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
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
            // acquired contents are undefined
            mImpl->mCurrentLayout = ImageLayout::kUndefined;
            return true;
        }
        return false;
    }

    bool Swapchain::BeginRendering(CommandList& cmd, const float clearColor[4], LoadOp loadOp) {
        if (mImpl == nullptr || mImpl->mCurrentImage >= mImpl->mImages.size()) {
            return false;
        }

        // current layout (Undefined after acquire, PresentSrc after a
        // previous EndRendering) -> color attachment
        SyncInfo toColor{};
        toColor.mSrcStage = mImpl->mCurrentLayout == ImageLayout::kPresentSrc
                ? PipelineStage::kBottomOfPipe
                : PipelineStage::kTopOfPipe;
        toColor.mSrcAccess = Access::kNone;
        toColor.mDstStage = PipelineStage::kColorAttachmentOutput;
        toColor.mDstAccess = Access::kColorAttachmentWrite;
        RecordImageBarrier(cmd.mImpl->mCommandBuffer, mImpl->mImages[mImpl->mCurrentImage],
                VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
                mImpl->mCurrentLayout, ImageLayout::kColorAttachment, toColor);
        mImpl->mCurrentLayout = ImageLayout::kColorAttachment;

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = mImpl->mImageViews[mImpl->mCurrentImage];
        colorAttachment.imageLayout = ToVkImageLayout(ImageLayout::kColorAttachment);
        colorAttachment.loadOp = loadOp == LoadOp::kClear
                ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
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
        SyncInfo toPresent{};
        toPresent.mSrcStage = PipelineStage::kColorAttachmentOutput;
        toPresent.mSrcAccess = Access::kColorAttachmentWrite;
        toPresent.mDstStage = PipelineStage::kBottomOfPipe;
        toPresent.mDstAccess = Access::kNone;
        RecordImageBarrier(cmd.mImpl->mCommandBuffer, mImpl->mImages[mImpl->mCurrentImage],
                VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
                ImageLayout::kColorAttachment, ImageLayout::kPresentSrc, toPresent);
        mImpl->mCurrentLayout = ImageLayout::kPresentSrc;
    }

    bool Swapchain::BeginTransfer(CommandList& cmd) {
        if (mImpl == nullptr || mImpl->mCurrentImage >= mImpl->mImages.size()) {
            return false;
        }
        SyncInfo toTransfer{};
        toTransfer.mSrcStage = mImpl->mCurrentLayout == ImageLayout::kPresentSrc
                ? PipelineStage::kBottomOfPipe
                : PipelineStage::kTopOfPipe;
        toTransfer.mSrcAccess = Access::kNone;
        toTransfer.mDstStage = PipelineStage::kTransfer;
        toTransfer.mDstAccess = Access::kTransferWrite;
        RecordImageBarrier(cmd.mImpl->mCommandBuffer, mImpl->mImages[mImpl->mCurrentImage],
                VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
                mImpl->mCurrentLayout, ImageLayout::kTransferDst, toTransfer);
        mImpl->mCurrentLayout = ImageLayout::kTransferDst;
        return true;
    }

    void Swapchain::EndTransfer(CommandList& cmd) {
        if (mImpl == nullptr || mImpl->mCurrentLayout != ImageLayout::kTransferDst) {
            return;
        }
        SyncInfo toPresent{};
        toPresent.mSrcStage = PipelineStage::kTransfer;
        toPresent.mSrcAccess = Access::kTransferWrite;
        toPresent.mDstStage = PipelineStage::kBottomOfPipe;
        toPresent.mDstAccess = Access::kNone;
        RecordImageBarrier(cmd.mImpl->mCommandBuffer, mImpl->mImages[mImpl->mCurrentImage],
                VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
                ImageLayout::kTransferDst, ImageLayout::kPresentSrc, toPresent);
        mImpl->mCurrentLayout = ImageLayout::kPresentSrc;
    }

    bool Swapchain::Present(CommandList& cmd) {
        if (mImpl == nullptr || mImpl->mSwapchain == VK_NULL_HANDLE
                || mImpl->mCurrentImage >= mImpl->mRenderFinished.size()) {
            return false;
        }

        VkSemaphore renderFinished = mImpl->mRenderFinished[mImpl->mCurrentImage];

        // vkQueueSubmit's wait-stage mask uses the 32-bit VkPipelineStageFlags
        // (not the 64-bit VkPipelineStageFlagBits2 used by synchronization2).
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
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