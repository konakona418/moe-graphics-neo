#include "RHI/Swapchain.hpp"
#include <Core/Profile.hpp>

#include "RHI/CommandList.hpp"
#include "RHI/Image.hpp"
#include "RHI/TimelineSemaphore.hpp"
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
        if (mImpl) {
            mImpl->mMsaaImage.Destroy();
        }
        mImpl.reset();
    }

    bool Swapchain::AcquireImage() {
        MOE_PROFILE_ZONE();
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

    bool Swapchain::BeginRendering(CommandList& cmd, const float clearColor[4], LoadOp loadOp,
            const Image* depthImage, float depthClear, LoadOp depthLoadOp) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || mImpl->mCurrentImage >= mImpl->mImages.size()) {
            return false;
        }

        // current layout (Undefined after acquire, PresentSrc after a
        // previous EndRendering) -> color attachment. When multisampled this
        // image is the resolve target, so it must be in ColorAttachment layout
        // as well.
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

        // multisampled rendering: draw into the internal MS image and resolve
        // into the swapchain image at the end of the pass
        const bool msaa = mImpl->mSampleCount > 1 && mImpl->mMsaaImage.mImpl != nullptr;
        if (msaa) {
            SyncInfo toMsaa{};
            toMsaa.mSrcStage = mImpl->mMsaaLayout == ImageLayout::kColorAttachment
                    ? PipelineStage::kColorAttachmentOutput
                    : PipelineStage::kTopOfPipe;
            toMsaa.mSrcAccess = mImpl->mMsaaLayout == ImageLayout::kColorAttachment
                    ? Access::kColorAttachmentWrite
                    : Access::kNone;
            toMsaa.mDstStage = PipelineStage::kColorAttachmentOutput;
            toMsaa.mDstAccess = Access::kColorAttachmentWrite;
            RecordImageBarrier(cmd.mImpl->mCommandBuffer, mImpl->mMsaaImage.mImpl->mImage,
                    VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
                    mImpl->mMsaaLayout, ImageLayout::kColorAttachment, toMsaa);
            mImpl->mMsaaLayout = ImageLayout::kColorAttachment;
        }

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = msaa
                ? mImpl->mMsaaImage.mImpl->mView
                : mImpl->mImageViews[mImpl->mCurrentImage];
        colorAttachment.imageLayout = ToVkImageLayout(ImageLayout::kColorAttachment);
        colorAttachment.loadOp = loadOp == LoadOp::kClear
                ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {
                {clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
        if (msaa) {
            colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            colorAttachment.resolveImageView = mImpl->mImageViews[mImpl->mCurrentImage];
            colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkRenderingAttachmentInfo depthAttachment{};
        if (depthImage != nullptr) {
            depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAttachment.imageView = depthImage->mImpl->mView;
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp = depthLoadOp == LoadOp::kClear
                    ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachment.clearValue.depthStencil = {depthClear, 0};
        }

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = {{0, 0}, {mImpl->mWidth, mImpl->mHeight}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = depthImage != nullptr ? &depthAttachment : nullptr;
        vkCmdBeginRendering(cmd.mImpl->mCommandBuffer, &renderingInfo);
        return true;
    }

    void Swapchain::EndRendering(CommandList& cmd) {
        MOE_PROFILE_ZONE();
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
        MOE_PROFILE_ZONE();
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
        MOE_PROFILE_ZONE();
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

    bool Swapchain::Present(CommandList& cmd, std::span<const TimelineWait> waits) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || mImpl->mSwapchain == VK_NULL_HANDLE
                || mImpl->mCurrentImage >= mImpl->mRenderFinished.size()) {
            return false;
        }

        VkSemaphore renderFinished = mImpl->mRenderFinished[mImpl->mCurrentImage];

        std::vector<VkSemaphoreSubmitInfo> waitInfos;
        waitInfos.reserve(1 + waits.size());
        VkSemaphoreSubmitInfo imageAvailable{};
        imageAvailable.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        imageAvailable.semaphore = mImpl->mImageAvailable;
        imageAvailable.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        waitInfos.push_back(imageAvailable);
        for (const auto& wait : waits) {
            VkSemaphoreSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            info.semaphore = wait.mSemaphore->mImpl->mSemaphore;
            info.value = wait.mValue;
            info.stageMask = ToVkPipelineStage(wait.mStage);
            waitInfos.push_back(info);
        }

        VkSemaphoreSubmitInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signalInfo.semaphore = renderFinished;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        VkCommandBufferSubmitInfo commandInfo{};
        commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandInfo.commandBuffer = cmd.mImpl->mCommandBuffer;

        VkSubmitInfo2 submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submit.waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size());
        submit.pWaitSemaphoreInfos = waitInfos.data();
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &commandInfo;
        submit.signalSemaphoreInfoCount = 1;
        submit.pSignalSemaphoreInfos = &signalInfo;
        if (vkQueueSubmit2(mImpl->mDevice->mGraphicsQueue, 1, &submit, mImpl->mInFlight) != VK_SUCCESS) {
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

    uint32_t Swapchain::GetSampleCount() const {
        return mImpl ? mImpl->mSampleCount : 1;
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