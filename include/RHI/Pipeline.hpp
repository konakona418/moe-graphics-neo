#pragma once

#include "RHI/RHICommon.hpp"
#include "RHI/PipelineState.hpp"

#include <cstdint>
#include <memory>

namespace moe::rhi {
    class Device;
    class CommandList;
    class DescriptorSetLayout;
    struct PipelineNode;

    // Handle to a cached graphics pipeline. Non-owning: the pipeline (and its
    // Vulkan objects) are owned by the PipelineCache and live until the cache
    // is cleared or destroyed.
    class GraphicsPipeline {
    public:
        GraphicsPipeline();
        ~GraphicsPipeline();

        GraphicsPipeline(const GraphicsPipeline&) = delete;
        GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

        bool IsValid() const;

        // Returns the descriptor set layout for the given set, allocated
        // against the exact layout object used by this pipeline.
        bool GetDescriptorSetLayout(uint32_t setIndex, DescriptorSetLayout& outLayout) const;

    private:
        friend class Device;
        friend class CommandList;
        friend class DefaultPipelineCache;
        friend class BindlessSet;

        const PipelineNode* mNode{nullptr};
    };

    // Handle to a cached compute pipeline (same ownership rules as above).
    class ComputePipeline {
    public:
        ComputePipeline();
        ~ComputePipeline();

        ComputePipeline(const ComputePipeline&) = delete;
        ComputePipeline& operator=(const ComputePipeline&) = delete;

        bool IsValid() const;
        bool GetDescriptorSetLayout(uint32_t setIndex, DescriptorSetLayout& outLayout) const;
        void GetWorkgroupSize(uint32_t& outX, uint32_t& outY, uint32_t& outZ) const;

    private:
        friend class Device;
        friend class CommandList;
        friend class DefaultPipelineCache;
        friend class BindlessSet;

        const PipelineNode* mNode{nullptr};
    };
}// namespace moe::rhi