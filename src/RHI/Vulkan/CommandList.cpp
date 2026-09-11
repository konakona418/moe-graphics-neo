#include "RHI/CommandList.hpp"

#include "RHI/Buffer.hpp"
#include "RHI/DescriptorSet.hpp"
#include "RHI/Image.hpp"
#include "RHI/Pipeline.hpp"
#include "Mappings.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"
#include <Core/Profile.hpp>

#include <algorithm>
#include <utility>

namespace moe::rhi {
    CommandList::CommandList() = default;

    CommandList::~CommandList() {
        MOE_RHI_ASSERT(mImpl == nullptr, "CommandList leaked: Destroy() not called");
    }

    void CommandList::Destroy() {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mDevice && mImpl->mCommandBuffer != VK_NULL_HANDLE) {
            // Deferred: the command buffer may still be in flight when destroyed.
            DeferredDeletion deletion;
            deletion.mCommandBuffer = mImpl->mCommandBuffer;
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
        }
        mImpl.reset();
    }

    void CommandList::Begin() {
        MOE_PROFILE_ZONE();
        vkResetCommandBuffer(mImpl->mCommandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(mImpl->mCommandBuffer, &beginInfo);
        mImpl->mRecording = true;
    }

    void CommandList::End() {
        MOE_PROFILE_ZONE();
        vkEndCommandBuffer(mImpl->mCommandBuffer);
        mImpl->mRecording = false;
    }

    void CommandList::CopyBuffer(const Buffer& src, const Buffer& dst,
            uint64_t size, uint64_t srcOffset, uint64_t dstOffset) {
        MOE_PROFILE_ZONE();
        VkBufferCopy region{};
        region.srcOffset = srcOffset;
        region.dstOffset = dstOffset;
        region.size = size;
        vkCmdCopyBuffer(mImpl->mCommandBuffer, src.mImpl->mBuffer, dst.mImpl->mBuffer, 1, &region);
    }

    void CommandList::CopyImageToBuffer(const Image& image, const Buffer& dst) {
        MOE_PROFILE_ZONE();
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = image.mImpl->mFormat == Format::kD32Float
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {image.mImpl->mWidth, image.mImpl->mHeight, image.mImpl->mDepth};
        vkCmdCopyImageToBuffer(mImpl->mCommandBuffer, image.mImpl->mImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.mImpl->mBuffer, 1, &region);
    }

    void CommandList::CopyBufferToImage(const Buffer& src, const Image& dst,
            uint32_t mipLevel, uint32_t baseArrayLayer, uint32_t layerCount,
            uint32_t bufferOffset) {
        MOE_PROFILE_ZONE();
        VkBufferImageCopy region{};
        region.bufferOffset = bufferOffset;
        region.bufferRowLength = 0; // tightly packed
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = dst.mImpl->mFormat == Format::kD32Float
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = mipLevel;
        region.imageSubresource.baseArrayLayer = baseArrayLayer;
        region.imageSubresource.layerCount = layerCount;
        region.imageOffset = {0, 0, 0};
        // a mip level's extent is the base extent shifted down by its index
        region.imageExtent = {
                std::max(1u, dst.mImpl->mWidth >> mipLevel),
                std::max(1u, dst.mImpl->mHeight >> mipLevel),
                std::max(1u, dst.mImpl->mDepth >> mipLevel)};
        vkCmdCopyBufferToImage(mImpl->mCommandBuffer, src.mImpl->mBuffer,
                dst.mImpl->mImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    void CommandList::CopyImage(const Image& src, ImageLayout srcLayout,
            const Image& dst, ImageLayout dstLayout) {
        MOE_PROFILE_ZONE();
        VkImageCopy region{};
        region.srcSubresource.aspectMask = src.mImpl->mFormat == Format::kD32Float
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.mipLevel = 0;
        region.srcSubresource.baseArrayLayer = 0;
        region.srcSubresource.layerCount = 1;
        region.dstSubresource.aspectMask = dst.mImpl->mFormat == Format::kD32Float
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.mipLevel = 0;
        region.dstSubresource.baseArrayLayer = 0;
        region.dstSubresource.layerCount = 1;
        region.extent = {src.mImpl->mWidth, src.mImpl->mHeight, src.mImpl->mDepth};
        vkCmdCopyImage(mImpl->mCommandBuffer, src.mImpl->mImage, ToVkImageLayout(srcLayout),
                dst.mImpl->mImage, ToVkImageLayout(dstLayout), 1, &region);
    }

    void CommandList::BlitImage(const Image& src, ImageLayout srcLayout,
            const Image& dst, ImageLayout dstLayout, Filter filter) {
        MOE_PROFILE_ZONE();
        VkImageBlit region{};
        region.srcSubresource.aspectMask = src.mImpl->mFormat == Format::kD32Float
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.mipLevel = 0;
        region.srcSubresource.baseArrayLayer = 0;
        region.srcSubresource.layerCount = 1;
        region.srcOffsets[0] = {0, 0, 0};
        region.srcOffsets[1] = {
                static_cast<int32_t>(src.mImpl->mWidth),
                static_cast<int32_t>(src.mImpl->mHeight),
                static_cast<int32_t>(src.mImpl->mDepth)};
        region.dstSubresource.aspectMask = dst.mImpl->mFormat == Format::kD32Float
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.mipLevel = 0;
        region.dstSubresource.baseArrayLayer = 0;
        region.dstSubresource.layerCount = 1;
        region.dstOffsets[0] = {0, 0, 0};
        region.dstOffsets[1] = {
                static_cast<int32_t>(dst.mImpl->mWidth),
                static_cast<int32_t>(dst.mImpl->mHeight),
                static_cast<int32_t>(dst.mImpl->mDepth)};
        vkCmdBlitImage(mImpl->mCommandBuffer, src.mImpl->mImage, ToVkImageLayout(srcLayout),
                dst.mImpl->mImage, ToVkImageLayout(dstLayout), 1, &region, ToVkFilter(filter));
    }

    void CommandList::BeginRendering(const Image& color, const float clearColor[4],
            const Image* depth, float depthClear, LoadOp colorLoadOp, LoadOp depthLoadOp,
            const Image* resolveColor) {
        MOE_PROFILE_ZONE();
        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = color.mImpl->mView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = colorLoadOp == LoadOp::kClear
                ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {
                {clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
        if (resolveColor != nullptr) {
            colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            colorAttachment.resolveImageView = resolveColor->mImpl->mView;
            colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkRenderingAttachmentInfo depthAttachment{};
        if (depth != nullptr) {
            depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAttachment.imageView = depth->mImpl->mView;
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp = depthLoadOp == LoadOp::kClear
                    ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachment.clearValue.depthStencil = {depthClear, 0};
        }

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = {{0, 0}, {color.mImpl->mWidth, color.mImpl->mHeight}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = depth != nullptr ? &depthAttachment : nullptr;
        vkCmdBeginRendering(mImpl->mCommandBuffer, &renderingInfo);
    }

    void CommandList::EndRendering() {
        MOE_PROFILE_ZONE();
        vkCmdEndRendering(mImpl->mCommandBuffer);
    }

    uintptr_t CommandList::GetVulkanHandle() const {
        return mImpl != nullptr
                ? reinterpret_cast<uintptr_t>(mImpl->mCommandBuffer) : 0;
    }

    void CommandList::BindDescriptorSet(const GraphicsPipeline& pipeline, const DescriptorSet& set, uint32_t setIndex) {
        vkCmdBindDescriptorSets(mImpl->mCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline.mNode->mPipelineLayout, setIndex, 1, &set.mImpl->mSet, 0, nullptr);
    }

    void CommandList::BindDescriptorSet(const ComputePipeline& pipeline, const DescriptorSet& set, uint32_t setIndex) {
        vkCmdBindDescriptorSets(mImpl->mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.mNode->mPipelineLayout, setIndex, 1, &set.mImpl->mSet, 0, nullptr);
    }

    void CommandList::SetPushConstants(const GraphicsPipeline& pipeline, uint32_t offset, size_t size, const void* data) {
        vkCmdPushConstants(mImpl->mCommandBuffer, pipeline.mNode->mPipelineLayout,
                pipeline.mNode->mPushConstantStages, offset, static_cast<uint32_t>(size), data);
    }

    void CommandList::SetPushConstants(const ComputePipeline& pipeline, uint32_t offset, size_t size, const void* data) {
        vkCmdPushConstants(mImpl->mCommandBuffer, pipeline.mNode->mPipelineLayout,
                pipeline.mNode->mPushConstantStages, offset, static_cast<uint32_t>(size), data);
    }

    void CommandList::BindGraphicsPipeline(const GraphicsPipeline& pipeline) {
        MOE_PROFILE_ZONE();
        vkCmdBindPipeline(mImpl->mCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.mNode->mPipeline);
    }

    void CommandList::Dispatch(const ComputePipeline& pipeline, uint32_t x, uint32_t y, uint32_t z) {
        MOE_PROFILE_ZONE();
        vkCmdBindPipeline(mImpl->mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.mNode->mPipeline);
        vkCmdDispatch(mImpl->mCommandBuffer, x, y, z);
    }

    void CommandList::SetViewport(uint32_t width, uint32_t height) {
        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
        vkCmdSetViewport(mImpl->mCommandBuffer, 0, 1, &viewport);
        const VkRect2D scissor{{0, 0}, {width, height}};
        vkCmdSetScissor(mImpl->mCommandBuffer, 0, 1, &scissor);
    }

    void CommandList::SetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
        const VkRect2D scissor{{x, y}, {width, height}};
        vkCmdSetScissor(mImpl->mCommandBuffer, 0, 1, &scissor);
    }

    void CommandList::BindVertexBuffer(const Buffer& buffer, uint32_t binding) {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(mImpl->mCommandBuffer, binding, 1, &buffer.mImpl->mBuffer, &offset);
    }

    void CommandList::BindIndexBuffer(const Buffer& buffer) {
        vkCmdBindIndexBuffer(mImpl->mCommandBuffer, buffer.mImpl->mBuffer, 0, VK_INDEX_TYPE_UINT32);
    }

    void CommandList::Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
        MOE_PROFILE_ZONE();
        vkCmdDraw(mImpl->mCommandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
    }

    void CommandList::DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
            int32_t vertexOffset, uint32_t firstInstance) {
        MOE_PROFILE_ZONE();
        vkCmdDrawIndexed(mImpl->mCommandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }

    void CommandList::MemoryBarrier(const SyncInfo& sync) {
        VkMemoryBarrier2 barrier{};
        FillSync(barrier, sync);

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(mImpl->mCommandBuffer, &dependency);
    }

    void CommandList::BufferBarrier(const Buffer& buffer, const SyncInfo& sync) {
        VkBufferMemoryBarrier2 barrier{};
        FillSync(barrier, sync);
        barrier.buffer = buffer.mImpl->mBuffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(mImpl->mCommandBuffer, &dependency);
    }

    void CommandList::ImageBarrier(const Image& image, ImageLayout srcLayout, ImageLayout dstLayout,
            const SyncInfo& sync) {
        const VkImageAspectFlags aspect = image.mImpl->mFormat == Format::kD32Float
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        const uint32_t layers = image.mImpl->mType == ImageType::kCube ? 6 : image.mImpl->mLayerCount;
        RecordImageBarrier(mImpl->mCommandBuffer, image.mImpl->mImage, aspect,
                image.mImpl->mMipLevels, layers, srcLayout, dstLayout, sync);
    }
}// namespace moe::rhi