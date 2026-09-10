#include "RHI/Pipeline.hpp"

#include "RHI/Shader.hpp"
#include "RhiInternal.hpp"

namespace moe::rhi {
    GraphicsPipeline::GraphicsPipeline() = default;

    GraphicsPipeline::~GraphicsPipeline() = default;

    bool GraphicsPipeline::IsValid() const {
        return mNode != nullptr;
    }

    bool GraphicsPipeline::GetDescriptorSetLayout(uint32_t setIndex, DescriptorSetLayout& outLayout) const {
        if (mNode == nullptr || setIndex >= mNode->mSetLayouts.size()) {
            return false;
        }
        outLayout.mImpl->mSetLayout = mNode->mSetLayouts[setIndex];
        outLayout.mImpl->mBindings = mNode->mSetBindings[setIndex];
        return true;
    }

    ComputePipeline::ComputePipeline() = default;

    ComputePipeline::~ComputePipeline() = default;

    bool ComputePipeline::IsValid() const {
        return mNode != nullptr;
    }

    bool ComputePipeline::GetDescriptorSetLayout(uint32_t setIndex, DescriptorSetLayout& outLayout) const {
        if (mNode == nullptr || setIndex >= mNode->mSetLayouts.size()) {
            return false;
        }
        outLayout.mImpl->mSetLayout = mNode->mSetLayouts[setIndex];
        outLayout.mImpl->mBindings = mNode->mSetBindings[setIndex];
        return true;
    }

    void ComputePipeline::GetWorkgroupSize(uint32_t& outX, uint32_t& outY, uint32_t& outZ) const {
        const Shader* compute = mNode ? mNode->mComputeState.mProgram->GetStage(ShaderStage::kCompute) : nullptr;
        if (compute == nullptr) {
            outX = outY = outZ = 1;
            return;
        }
        const ShaderReflection& reflection = compute->GetReflection();
        outX = reflection.mWorkgroupSizeX;
        outY = reflection.mWorkgroupSizeY;
        outZ = reflection.mWorkgroupSizeZ;
    }
}// namespace moe::rhi