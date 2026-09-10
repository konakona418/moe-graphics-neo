#pragma once

#include "RHI/RHICommon.hpp"
#include "RhiAssert.hpp"

#include <volk.h>

// Backend-agnostic enum -> Vulkan conversion helpers shared across the
// Vulkan backend translation units. Unknown values are an invariant violation
// and abort (never silently mapped to a wrong value).

namespace moe::rhi {
    inline VkFormat ToVkFormat(Format format) {
        switch (format) {
            case Format::kUndefined: return VK_FORMAT_UNDEFINED;
            case Format::kR8G8B8A8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
            case Format::kR8G8B8A8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
            case Format::kB8G8R8A8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
            case Format::kB8G8R8A8Srgb: return VK_FORMAT_B8G8R8A8_SRGB;
            case Format::kR16G16Float: return VK_FORMAT_R16G16_SFLOAT;
            case Format::kR32Float: return VK_FORMAT_R32_SFLOAT;
            case Format::kR32Uint: return VK_FORMAT_R32_UINT;
            case Format::kR32G32Float: return VK_FORMAT_R32G32_SFLOAT;
            case Format::kR32G32B32Float: return VK_FORMAT_R32G32B32_SFLOAT;
            case Format::kR16G16B16A16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
            case Format::kR32G32B32A32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
            case Format::kD32Float: return VK_FORMAT_D32_SFLOAT;
        }
        MOE_RHI_ASSERT(false, "ToVkFormat: unhandled Format");
        return VK_FORMAT_UNDEFINED;
    }

    inline VkSampleCountFlagBits ToVkSampleCount(uint32_t samples) {
        switch (samples) {
            case 1: return VK_SAMPLE_COUNT_1_BIT;
            case 2: return VK_SAMPLE_COUNT_2_BIT;
            case 4: return VK_SAMPLE_COUNT_4_BIT;
            case 8: return VK_SAMPLE_COUNT_8_BIT;
            default: break;
        }
        MOE_RHI_ASSERT(false, "ToVkSampleCount: unsupported sample count");
        return VK_SAMPLE_COUNT_1_BIT;
    }

    inline VkDescriptorType ToVkDescriptorType(DescriptorType type) {
        switch (type) {
            case DescriptorType::kUniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            case DescriptorType::kStorageBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case DescriptorType::kCombinedImageSampler: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            case DescriptorType::kSampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            case DescriptorType::kStorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            case DescriptorType::kSampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
        }
        MOE_RHI_ASSERT(false, "ToVkDescriptorType: unhandled DescriptorType");
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }

    inline VkImageType ToVkImageType(ImageType type) {
        switch (type) {
            case ImageType::k2D: return VK_IMAGE_TYPE_2D;
            case ImageType::k3D: return VK_IMAGE_TYPE_3D;
            case ImageType::kCube: return VK_IMAGE_TYPE_2D;
        }
        MOE_RHI_ASSERT(false, "ToVkImageType: unhandled ImageType");
        return VK_IMAGE_TYPE_2D;
    }

    inline VkImageUsageFlags ToVkImageUsage(ImageUsage usage) {
        VkImageUsageFlags flags = 0;
        if (HasFlag(usage, ImageUsage::kSampled)) {
            flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if (HasFlag(usage, ImageUsage::kStorage)) {
            flags |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if (HasFlag(usage, ImageUsage::kColorAttachment)) {
            flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (HasFlag(usage, ImageUsage::kDepthAttachment)) {
            flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }
        if (HasFlag(usage, ImageUsage::kTransferSrc)) {
            flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }
        if (HasFlag(usage, ImageUsage::kTransferDst)) {
            flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }
        return flags;
    }

    inline VkImageLayout ToVkImageLayout(ImageLayout layout) {
        switch (layout) {
            case ImageLayout::kUndefined: return VK_IMAGE_LAYOUT_UNDEFINED;
            case ImageLayout::kGeneral: return VK_IMAGE_LAYOUT_GENERAL;
            case ImageLayout::kColorAttachment: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            case ImageLayout::kDepthStencilAttachment: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            case ImageLayout::kShaderReadOnly: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            case ImageLayout::kTransferSrc: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            case ImageLayout::kTransferDst: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            case ImageLayout::kPresentSrc: return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }
        MOE_RHI_ASSERT(false, "ToVkImageLayout: unhandled ImageLayout");
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }

    inline VkFilter ToVkFilter(Filter filter) {
        switch (filter) {
            case Filter::kNearest: return VK_FILTER_NEAREST;
            case Filter::kLinear: return VK_FILTER_LINEAR;
        }
        MOE_RHI_ASSERT(false, "ToVkFilter: unhandled Filter");
        return VK_FILTER_NEAREST;
    }

    inline VkSamplerAddressMode ToVkAddressMode(AddressMode mode) {
        switch (mode) {
            case AddressMode::kRepeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case AddressMode::kMirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case AddressMode::kClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case AddressMode::kClampToBorder: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        }
        MOE_RHI_ASSERT(false, "ToVkAddressMode: unhandled AddressMode");
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }

    inline VkShaderStageFlagBits ToVkShaderStage(ShaderStage stage) {
        switch (stage) {
            case ShaderStage::kVertex: return VK_SHADER_STAGE_VERTEX_BIT;
            case ShaderStage::kFragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
            case ShaderStage::kGeometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
            case ShaderStage::kCompute: return VK_SHADER_STAGE_COMPUTE_BIT;
        }
        MOE_RHI_ASSERT(false, "ToVkShaderStage: unhandled ShaderStage");
        return VK_SHADER_STAGE_COMPUTE_BIT;
    }

    inline VkPipelineStageFlags2 ToVkPipelineStage(PipelineStage stage) {
        switch (stage) {
            case PipelineStage::kTopOfPipe: return VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            case PipelineStage::kDrawIndirect: return VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            case PipelineStage::kVertexInput: return VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            case PipelineStage::kVertexShader: return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            case PipelineStage::kGeometryShader: return VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
            case PipelineStage::kFragmentShader: return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            case PipelineStage::kEarlyFragmentTests: return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            case PipelineStage::kLateFragmentTests: return VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            case PipelineStage::kColorAttachmentOutput: return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            case PipelineStage::kComputeShader: return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            case PipelineStage::kTransfer: return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            case PipelineStage::kHost: return VK_PIPELINE_STAGE_2_HOST_BIT;
            case PipelineStage::kBottomOfPipe: return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        }
        MOE_RHI_ASSERT(false, "ToVkPipelineStage: unhandled PipelineStage");
        return 0;
    }

    inline VkAccessFlags2 ToVkAccess(Access access) {
        switch (access) {
            case Access::kNone: return VK_ACCESS_2_NONE;
            case Access::kIndirectCommandRead: return VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            case Access::kIndexRead: return VK_ACCESS_2_INDEX_READ_BIT;
            case Access::kVertexAttributeRead: return VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            case Access::kUniformRead: return VK_ACCESS_2_UNIFORM_READ_BIT;
            case Access::kInputAttachmentRead: return VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
            case Access::kShaderRead: return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            case Access::kShaderWrite: return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            case Access::kColorAttachmentRead: return VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
            case Access::kColorAttachmentWrite: return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            case Access::kDepthStencilAttachmentRead: return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            case Access::kDepthStencilAttachmentWrite: return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            case Access::kTransferRead: return VK_ACCESS_2_TRANSFER_READ_BIT;
            case Access::kTransferWrite: return VK_ACCESS_2_TRANSFER_WRITE_BIT;
            case Access::kHostRead: return VK_ACCESS_2_HOST_READ_BIT;
            case Access::kHostWrite: return VK_ACCESS_2_HOST_WRITE_BIT;
            case Access::kMemoryRead: return VK_ACCESS_2_MEMORY_READ_BIT;
            case Access::kMemoryWrite: return VK_ACCESS_2_MEMORY_WRITE_BIT;
        }
        MOE_RHI_ASSERT(false, "ToVkAccess: unhandled Access");
        return 0;
    }

    // ---- barrier recording (single place that fills the sync fields) ----

    inline void FillSync(VkMemoryBarrier2& barrier, const SyncInfo& sync) {
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = ToVkPipelineStage(sync.mSrcStage);
        barrier.srcAccessMask = ToVkAccess(sync.mSrcAccess);
        barrier.dstStageMask = ToVkPipelineStage(sync.mDstStage);
        barrier.dstAccessMask = ToVkAccess(sync.mDstAccess);
    }

    inline void FillSync(VkBufferMemoryBarrier2& barrier, const SyncInfo& sync) {
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = ToVkPipelineStage(sync.mSrcStage);
        barrier.srcAccessMask = ToVkAccess(sync.mSrcAccess);
        barrier.dstStageMask = ToVkPipelineStage(sync.mDstStage);
        barrier.dstAccessMask = ToVkAccess(sync.mDstAccess);
    }

    inline void FillSync(VkImageMemoryBarrier2& barrier, const SyncInfo& sync) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = ToVkPipelineStage(sync.mSrcStage);
        barrier.srcAccessMask = ToVkAccess(sync.mSrcAccess);
        barrier.dstStageMask = ToVkPipelineStage(sync.mDstStage);
        barrier.dstAccessMask = ToVkAccess(sync.mDstAccess);
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }

    // Records one image layout transition for a raw VkImage (swapchain
    // images included; the RHI Image wrapper feeds its own view info here).
    inline void RecordImageBarrier(VkCommandBuffer cmd, VkImage image,
            VkImageAspectFlags aspect, uint32_t levelCount, uint32_t layerCount,
            ImageLayout oldLayout, ImageLayout newLayout, const SyncInfo& sync) {
        VkImageMemoryBarrier2 barrier{};
        FillSync(barrier, sync);
        barrier.oldLayout = ToVkImageLayout(oldLayout);
        barrier.newLayout = ToVkImageLayout(newLayout);
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspect;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = levelCount;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = layerCount;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
    }
}// namespace moe::rhi