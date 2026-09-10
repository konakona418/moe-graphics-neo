#include "RHI/BindlessSet.hpp"

#include "Core/Error.hpp"
#include "RHI/CommandList.hpp"
#include "RHI/Device.hpp"
#include "RHI/Image.hpp"
#include "RHI/Pipeline.hpp"
#include "RHI/Sampler.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

namespace moe::rhi {
    struct BindlessSet::Impl {
        DeviceImpl* mDevice{nullptr};
        VkDescriptorPool mPool{VK_NULL_HANDLE};
        VkDescriptorSetLayout mLayout{VK_NULL_HANDLE};
        VkDescriptorSet mSet{VK_NULL_HANDLE};
        Sampler mDefaultNearest;
        Sampler mDefaultLinear;
        bool mInitialized{false};
    };

    BindlessSet::BindlessSet() = default;

    BindlessSet::~BindlessSet() {
        MOE_RHI_ASSERT(mImpl == nullptr, "BindlessSet leaked: Destroy() not called");
    }

    bool BindlessSet::Init(Device& device) {
        if (mImpl != nullptr) {
            return true; // already initialized
        }
        mImpl = std::make_unique<BindlessSet::Impl>();
        mImpl->mDevice = device.mImpl.get();

        // pool: update-after-bind (both bindings are written in place at
        // runtime indices, exactly like the old engine's VulkanBindlessSet)
        const VkDescriptorPoolSize poolSizes[2] = {
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxImages},
                {VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSamplers},
        };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        if (vkCreateDescriptorPool(mImpl->mDevice->mDevice, &poolInfo, nullptr, &mImpl->mPool)
                != VK_SUCCESS) {
            moe::Error::Set("BindlessSet: descriptor pool creation failed");
            mImpl.reset();
            return false;
        }

        const VkDescriptorSetLayoutBinding bindings[2] = {
                {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxImages, VK_SHADER_STAGE_ALL, nullptr},
                {1, VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSamplers, VK_SHADER_STAGE_ALL, nullptr},
        };
        const VkDescriptorBindingFlags bindingFlags[2] = {
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flagsInfo.bindingCount = 2;
        flagsInfo.pBindingFlags = bindingFlags;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.pNext = &flagsInfo;
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(mImpl->mDevice->mDevice, &layoutInfo, nullptr,
                    &mImpl->mLayout) != VK_SUCCESS) {
            moe::Error::Set("BindlessSet: layout creation failed");
            vkDestroyDescriptorPool(mImpl->mDevice->mDevice, mImpl->mPool, nullptr);
            mImpl.reset();
            return false;
        }

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = mImpl->mPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &mImpl->mLayout;
        if (vkAllocateDescriptorSets(mImpl->mDevice->mDevice, &allocInfo, &mImpl->mSet)
                != VK_SUCCESS) {
            moe::Error::Set("BindlessSet: set allocation failed");
            vkDestroyDescriptorSetLayout(mImpl->mDevice->mDevice, mImpl->mLayout, nullptr);
            vkDestroyDescriptorPool(mImpl->mDevice->mDevice, mImpl->mPool, nullptr);
            mImpl.reset();
            return false;
        }

        // default samplers (nearest @0, linear @1) — the old engine registered
        // them at init; shaders address them by these fixed ids
        mImpl->mInitialized = true; // AddSampler checks it; set before use
        rhi::SamplerCreateInfo nearestInfo{};
        nearestInfo.mMinFilter = rhi::Filter::kNearest;
        nearestInfo.mMagFilter = rhi::Filter::kNearest;
        if (!device.CreateSampler(nearestInfo, mImpl->mDefaultNearest)
                || !AddSampler(0, mImpl->mDefaultNearest)) {
            moe::Error::Set("BindlessSet: default nearest sampler failed");
            Destroy();
            return false;
        }
        rhi::SamplerCreateInfo linearInfo{};
        linearInfo.mMinFilter = rhi::Filter::kLinear;
        linearInfo.mMagFilter = rhi::Filter::kLinear;
        if (!device.CreateSampler(linearInfo, mImpl->mDefaultLinear)
                || !AddSampler(1, mImpl->mDefaultLinear)) {
            moe::Error::Set("BindlessSet: default linear sampler failed");
            Destroy();
            return false;
        }

        return true;
    }

    void BindlessSet::Destroy() {
        if (mImpl == nullptr) {
            return;
        }
        mImpl->mDefaultNearest.Destroy();
        mImpl->mDefaultLinear.Destroy();
        if (mImpl->mDevice != nullptr) {
            if (mImpl->mLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(mImpl->mDevice->mDevice, mImpl->mLayout, nullptr);
            }
            if (mImpl->mPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(mImpl->mDevice->mDevice, mImpl->mPool, nullptr);
            }
        }
        mImpl.reset();
    }

    bool BindlessSet::AddImage(uint32_t id, const Image& image) {
        if (mImpl == nullptr || !mImpl->mInitialized || id >= kMaxImages
                || image.mImpl == nullptr || image.mImpl->mView == VK_NULL_HANDLE) {
            return false;
        }
        const VkDescriptorImageInfo imageInfo{
                VK_NULL_HANDLE,
                image.mImpl->mView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet write{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                mImpl->mSet,
                0, // image binding
                id,
                1,
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                &imageInfo,
                nullptr,
                nullptr,
        };
        vkUpdateDescriptorSets(mImpl->mDevice->mDevice, 1, &write, 0, nullptr);
        return true;
    }

    bool BindlessSet::AddSampler(uint32_t id, const Sampler& sampler) {
        if (mImpl == nullptr || !mImpl->mInitialized || id >= kMaxSamplers
                || sampler.mImpl == nullptr || sampler.mImpl->mSampler == VK_NULL_HANDLE) {
            return false;
        }
        const VkDescriptorImageInfo imageInfo{
                sampler.mImpl->mSampler,
                VK_NULL_HANDLE,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet write{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                mImpl->mSet,
                1, // sampler binding
                id,
                1,
                VK_DESCRIPTOR_TYPE_SAMPLER,
                &imageInfo,
                nullptr,
                nullptr,
        };
        vkUpdateDescriptorSets(mImpl->mDevice->mDevice, 1, &write, 0, nullptr);
        return true;
    }

    uint32_t BindlessSet::GetImageCapacity() const {
        return kMaxImages;
    }

    uint32_t BindlessSet::GetSamplerCapacity() const {
        return kMaxSamplers;
    }

    bool BindlessSet::IsValid() const {
        return mImpl != nullptr && mImpl->mInitialized;
    }

    void BindlessSet::Bind(CommandList& cmd, const GraphicsPipeline& pipeline, uint32_t setIndex) {
        if (mImpl == nullptr || !mImpl->mInitialized) {
            return;
        }
        vkCmdBindDescriptorSets(cmd.mImpl->mCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline.mNode->mPipelineLayout, setIndex, 1, &mImpl->mSet, 0, nullptr);
    }

    void BindlessSet::Bind(CommandList& cmd, const ComputePipeline& pipeline, uint32_t setIndex) {
        if (mImpl == nullptr || !mImpl->mInitialized) {
            return;
        }
        vkCmdBindDescriptorSets(cmd.mImpl->mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.mNode->mPipelineLayout, setIndex, 1, &mImpl->mSet, 0, nullptr);
    }
}// namespace moe::rhi
