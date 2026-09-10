#pragma once

#include "RHI/RHICommon.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace moe::rhi {
    class Device;
    class Buffer;
    class Image;
    class Sampler;
    class CommandList;
    struct DescriptorSetLayoutImpl;
    struct DescriptorSetImpl;

    // Handle to a descriptor set layout (owned by the pipeline node it came
    // from). Descriptor sets allocated against it use the exact same Vulkan
    // layout object as the pipeline, so binding is always compatible.
    class DescriptorSetLayout {
    public:
        DescriptorSetLayout();
        ~DescriptorSetLayout();

        DescriptorSetLayout(DescriptorSetLayout&&) noexcept;
        DescriptorSetLayout& operator=(DescriptorSetLayout&&) noexcept;
        DescriptorSetLayout(const DescriptorSetLayout&) = delete;
        DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

        bool IsValid() const;
        const std::vector<DescriptorBindingInfo>& GetBindings() const;

    private:
        friend class Device;
        friend class PipelineNode;
        friend class GraphicsPipeline;
        friend class ComputePipeline;
        friend class DefaultPipelineCache;

        std::unique_ptr<DescriptorSetLayoutImpl> mImpl;
    };

    // A descriptor set allocated from the device's internal descriptor pool.
    // Bindings are written in place and the set is bound per draw/dispatch.
    class DescriptorSet {
    public:
        DescriptorSet();
        ~DescriptorSet();

        DescriptorSet(const DescriptorSet&) = delete;
        DescriptorSet& operator=(const DescriptorSet&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the set was
        // created but not destroyed (leak trap).
        void Destroy();

        // Points binding at the given buffer (usage must match the binding's
        // descriptor type: kUniformBuffer or kStorageBuffer).
        bool WriteBuffer(uint32_t binding, const Buffer& buffer);

        // Points binding at the given image (type must be kStorageImage or
        // kSampledImage). The image must be in the layout implied by the type
        // (General for storage, ShaderReadOnly for sampled) when the set is used.
        bool WriteImage(uint32_t binding, const Image& image, DescriptorType type);

        // Points binding at the given sampler (type must be kSampler). Used to
        // bind a sampler separately from an image (e.g. Texture3D + SamplerState).
        bool WriteSampler(uint32_t binding, const Sampler& sampler);

    private:
        friend class Device;
        friend class CommandList;

        std::unique_ptr<DescriptorSetImpl> mImpl;
    };
}// namespace moe::rhi