#pragma once

#include "RHI/Device.hpp"
#include "RHI/Buffer.hpp"
#include "RHI/CommandList.hpp"
#include "RHI/Image.hpp"
#include "RHI/Shader.hpp"
#include "RHI/DescriptorSet.hpp"
#include "RHI/Pipeline.hpp"
#include "RHI/PipelineState.hpp"
#include "RHI/Swapchain.hpp"

#include <array>
#include <string>
#include <vector>

#include <volk.h>
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>

// Backend implementation structs shared between RHI translation units. All
// Vulkan types stay inside src/RHI; the public headers never see them.

namespace moe::rhi {
    struct ShaderImpl {
        std::string mPath;
        ShaderStage mStage{ShaderStage::kCompute};
        std::vector<char> mCode; // .spv bytes
        ShaderReflection mReflection;
        uint64_t mContentHash{0};
    };

    struct ShaderProgramImpl {
        std::array<const Shader*, 4> mStages{}; // indexed by ShaderStage
        uint32_t mStageCount{0};
    };

    struct DescriptorSetLayoutImpl {
        VkDescriptorSetLayout mSetLayout{VK_NULL_HANDLE};
        std::vector<DescriptorBindingInfo> mBindings;
    };

    struct BufferImpl {
        VkBuffer mBuffer{VK_NULL_HANDLE};
        VmaAllocation mAllocation{VK_NULL_HANDLE};
        VkDeviceAddress mDeviceAddress{0};
        uint64_t mSize{0};
        bool mCpuVisible{false};
        BufferUsage mUsage{BufferUsage::kStorage};
        DeviceImpl* mDevice{nullptr};
    };

    struct ImageImpl {
        VkImage mImage{VK_NULL_HANDLE};
        VkImageView mView{VK_NULL_HANDLE};
        VmaAllocation mAllocation{VK_NULL_HANDLE};
        DeviceImpl* mDevice{nullptr};
        ImageType mType{ImageType::k2D};
        uint32_t mWidth{1};
        uint32_t mHeight{1};
        uint32_t mDepth{1};
        uint32_t mMipLevels{1};
        uint32_t mLayerCount{1};
        Format mFormat{Format::kR8G8B8A8Unorm};
        ImageUsage mUsage{ImageUsage::kSampled};
        // False for borrowed views (e.g. a swapchain image wrapper): Destroy()
        // then only drops the wrapper, never the underlying image/view.
        bool mOwned{true};
    };

    struct SamplerImpl {
        VkSampler mSampler{VK_NULL_HANDLE};
        DeviceImpl* mDevice{nullptr};
    };

    struct TimelineSemaphoreImpl {
        VkSemaphore mSemaphore{VK_NULL_HANDLE};
        DeviceImpl* mDevice{nullptr};
    };

    struct FenceImpl {
        VkFence mFence{VK_NULL_HANDLE};
        DeviceImpl* mDevice{nullptr};
    };

    struct DescriptorSetImpl {
        VkDescriptorSet mSet{VK_NULL_HANDLE};
        VkDescriptorPool mPool{VK_NULL_HANDLE};
        DeviceImpl* mDevice{nullptr};
    };

    // One entry in the device's deferred deletion queue. Only the fields that
    // were set are destroyed when flushed.
    struct DeferredDeletion {
        VkPipeline mPipeline{VK_NULL_HANDLE};
        VkPipelineLayout mPipelineLayout{VK_NULL_HANDLE};
        std::vector<VkDescriptorSetLayout> mSetLayouts;
        VkDescriptorPool mDescriptorPool{VK_NULL_HANDLE};
        VkBuffer mBuffer{VK_NULL_HANDLE};
        VmaAllocation mAllocation{VK_NULL_HANDLE};
        VkCommandBuffer mCommandBuffer{VK_NULL_HANDLE};
        VkCommandPool mCommandBufferPool{VK_NULL_HANDLE}; // pool the buffer came from
        VkImage mImage{VK_NULL_HANDLE};
        VkImageView mImageView{VK_NULL_HANDLE};
        VkSampler mSampler{VK_NULL_HANDLE};
        VkSemaphore mSemaphore{VK_NULL_HANDLE};
        VkFence mFence{VK_NULL_HANDLE};
    };

    // A cached pipeline node owned by the PipelineCache. GraphicsPipeline /
    // ComputePipeline are non-owning handles into it.
    struct PipelineNode {
        bool mIsCompute{false};
        GraphicsPipelineState mGraphicsState;
        ComputePipelineState mComputeState;
        VkPipeline mPipeline{VK_NULL_HANDLE};
        VkPipelineLayout mPipelineLayout{VK_NULL_HANDLE};
        VkShaderStageFlags mPushConstantStages{0};
        std::vector<VkDescriptorSetLayout> mSetLayouts;
        std::vector<std::vector<DescriptorBindingInfo>> mSetBindings;
        uint64_t mKeyHash{0};
    };

    struct DeviceImpl {
        vkb::Instance mInstance{};
        VkPhysicalDevice mPhysicalDevice{VK_NULL_HANDLE};
        VkDevice mDevice{VK_NULL_HANDLE};
        VkQueue mGraphicsQueue{VK_NULL_HANDLE};
        uint32_t mGraphicsQueueFamily{0};
        VkQueue mComputeQueue{VK_NULL_HANDLE};
        uint32_t mComputeQueueFamily{0};
        // highest sample count supported for both color and depth attachments
        uint32_t mMaxSampleCount{1};
        // cached combined depth-stencil format (see Device::GetDepthStencilFormat)
        mutable Format mDepthStencilFormat{Format::kUndefined};
        // one command pool per queue family (a pool is tied to one family)
        std::vector<VkCommandPool> mCommandPools;
        VmaAllocator mAllocator{VK_NULL_HANDLE};
        std::vector<DeferredDeletion> mDeferredDeletions;

        // Creates (once) the command pool for `family` and returns it, or
        // VK_NULL_HANDLE on failure. Never hardcodes a family index.
        VkCommandPool GetOrCreateCommandPool(uint32_t family) {
            if (family >= mCommandPools.size()) {
                mCommandPools.resize(family + 1, VK_NULL_HANDLE);
            }
            if (mCommandPools[family] == VK_NULL_HANDLE) {
                VkCommandPoolCreateInfo poolInfo{};
                poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                poolInfo.queueFamilyIndex = family;
                if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPools[family])
                        != VK_SUCCESS) {
                    return VK_NULL_HANDLE;
                }
            }
            return mCommandPools[family];
        }

        void EnqueueDeferred(DeferredDeletion deletion) {
            mDeferredDeletions.push_back(std::move(deletion));
        }

        void FlushDeferredDeletions() {
            for (auto& d : mDeferredDeletions) {
                if (d.mPipeline != VK_NULL_HANDLE) {
                    vkDestroyPipeline(mDevice, d.mPipeline, nullptr);
                }
                if (d.mPipelineLayout != VK_NULL_HANDLE) {
                    vkDestroyPipelineLayout(mDevice, d.mPipelineLayout, nullptr);
                }
                for (auto layout : d.mSetLayouts) {
                    vkDestroyDescriptorSetLayout(mDevice, layout, nullptr);
                }
                if (d.mDescriptorPool != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(mDevice, d.mDescriptorPool, nullptr);
                }
                if (d.mBuffer != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(mAllocator, d.mBuffer, d.mAllocation);
                }
                if (d.mCommandBuffer != VK_NULL_HANDLE && d.mCommandBufferPool != VK_NULL_HANDLE) {
                    vkFreeCommandBuffers(mDevice, d.mCommandBufferPool, 1, &d.mCommandBuffer);
                }
                if (d.mImageView != VK_NULL_HANDLE) {
                    vkDestroyImageView(mDevice, d.mImageView, nullptr);
                }
                if (d.mImage != VK_NULL_HANDLE) {
                    vmaDestroyImage(mAllocator, d.mImage, d.mAllocation);
                }
                if (d.mSampler != VK_NULL_HANDLE) {
                    vkDestroySampler(mDevice, d.mSampler, nullptr);
                }
                if (d.mSemaphore != VK_NULL_HANDLE) {
                    vkDestroySemaphore(mDevice, d.mSemaphore, nullptr);
                }
                if (d.mFence != VK_NULL_HANDLE) {
                    vkDestroyFence(mDevice, d.mFence, nullptr);
                }
            }
            mDeferredDeletions.clear();
        }
    };

    struct CommandListImpl {
        VkCommandBuffer mCommandBuffer{VK_NULL_HANDLE};
        VkCommandPool mPool{VK_NULL_HANDLE};
        bool mRecording{false};
        DeviceImpl* mDevice{nullptr};
    };

    struct SwapchainImpl {
        VkSurfaceKHR mSurface{VK_NULL_HANDLE};
        VkSwapchainKHR mSwapchain{VK_NULL_HANDLE};
        std::vector<VkImage> mImages;
        std::vector<VkImageView> mImageViews;
        VkSemaphore mImageAvailable{VK_NULL_HANDLE}; // one per frame (single-frame-in-flight)
        // one render-finished semaphore per swapchain image: it must not be
        // re-signaled while the swapchain still uses that image's present
        std::vector<VkSemaphore> mRenderFinished;
        VkFence mInFlight{VK_NULL_HANDLE};
        uint32_t mCurrentImage{0};
        uint32_t mWidth{0};
        uint32_t mHeight{0};
        Format mFormat{Format::kR8G8B8A8Unorm};
        // The swapchain owns its image's layout state: BeginRendering /
        // EndRendering keep it in sync so callers never hand-write the
        // PresentSrc <-> ColorAttachment transitions.
        ImageLayout mCurrentLayout{ImageLayout::kUndefined};
        // Multisampled color image (valid when mSampleCount > 1): passes render
        // into it and resolve into the acquired swapchain image.
        Image mMsaaImage;
        uint32_t mSampleCount{1};
        ImageLayout mMsaaLayout{ImageLayout::kUndefined};
        DeviceImpl* mDevice{nullptr};
    };
}// namespace moe::rhi