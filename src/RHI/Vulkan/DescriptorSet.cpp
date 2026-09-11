#include "RHI/DescriptorSet.hpp"
#include <Core/Profile.hpp>

#include "RHI/Buffer.hpp"
#include "RHI/Image.hpp"
#include "RHI/Sampler.hpp"
#include "Mappings.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <utility>

namespace moe::rhi {
    DescriptorSetLayout::DescriptorSetLayout() : mImpl(std::make_unique<DescriptorSetLayoutImpl>()) {}

    DescriptorSetLayout::~DescriptorSetLayout() = default;

    DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout&&) noexcept = default;

    DescriptorSetLayout& DescriptorSetLayout::operator=(DescriptorSetLayout&&) noexcept = default;

    bool DescriptorSetLayout::IsValid() const {
        return mImpl && mImpl->mSetLayout != VK_NULL_HANDLE;
    }

    const std::vector<DescriptorBindingInfo>& DescriptorSetLayout::GetBindings() const {
        return mImpl->mBindings;
    }

    DescriptorSet::DescriptorSet() = default;

    DescriptorSet::~DescriptorSet() {
        MOE_RHI_ASSERT(mImpl == nullptr, "DescriptorSet leaked: Destroy() not called");
    }

    void DescriptorSet::Destroy() {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mDevice && mImpl->mPool != VK_NULL_HANDLE) {
            DeferredDeletion deletion;
            deletion.mDescriptorPool = mImpl->mPool;
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
        }
        mImpl.reset();
    }

    bool DescriptorSet::WriteBuffer(uint32_t binding, const Buffer& buffer) {
        MOE_PROFILE_ZONE();
        if (!mImpl || !mImpl->mDevice || mImpl->mSet == VK_NULL_HANDLE) {
            return false;
        }
        if (buffer.mImpl == nullptr || buffer.mImpl->mBuffer == VK_NULL_HANDLE) {
            return false;
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffer.mImpl->mBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;

        // The buffer must have been created with a uniform or storage usage;
        // anything else would silently bind a wrong descriptor type.
        const bool isUniform = HasFlag(buffer.mImpl->mUsage, BufferUsage::kUniform);
        const bool isStorage = HasFlag(buffer.mImpl->mUsage, BufferUsage::kStorage);
        MOE_RHI_ASSERT(isUniform || isStorage, "WriteBuffer: buffer must have uniform or storage usage");

        VkDescriptorType type = isUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = mImpl->mSet;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = type;
        write.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(mImpl->mDevice->mDevice, 1, &write, 0, nullptr);
        return true;
    }

    bool DescriptorSet::WriteImage(uint32_t binding, const Image& image, DescriptorType type) {
        MOE_PROFILE_ZONE();
        if (!mImpl || !mImpl->mDevice || mImpl->mSet == VK_NULL_HANDLE) {
            return false;
        }
        if (!image.mImpl || image.mImpl->mView == VK_NULL_HANDLE) {
            return false;
        }
        MOE_RHI_ASSERT(type == DescriptorType::kStorageImage || type == DescriptorType::kSampledImage,
                "WriteImage: type must be kStorageImage or kSampledImage");

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = image.mImpl->mView;
        imageInfo.imageLayout = type == DescriptorType::kStorageImage
                ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = mImpl->mSet;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = ToVkDescriptorType(type);
        write.pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(mImpl->mDevice->mDevice, 1, &write, 0, nullptr);
        return true;
    }

    bool DescriptorSet::WriteSampler(uint32_t binding, const Sampler& sampler) {
        MOE_PROFILE_ZONE();
        if (!mImpl || !mImpl->mDevice || mImpl->mSet == VK_NULL_HANDLE) {
            return false;
        }
        if (!sampler.mImpl || sampler.mImpl->mSampler == VK_NULL_HANDLE) {
            return false;
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = sampler.mImpl->mSampler;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = mImpl->mSet;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(mImpl->mDevice->mDevice, 1, &write, 0, nullptr);
        return true;
    }
}// namespace moe::rhi