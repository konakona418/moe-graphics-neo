#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace moe::rhi {
    class Device;
    class Image;
    class Sampler;
    class CommandList;
    class GraphicsPipeline;
    class ComputePipeline;

    // Bindless descriptor set (descriptor indexing): one set holding a large
    // sampled-image array (binding 0) + sampler array (binding 1), written in
    // place at runtime indices. Requires DeviceCreateInfo::mEnableDescriptorIndexing.
    // Mirrors the old engine's VulkanBindlessSet; shaders address it with
    // unbounded arrays (e.g. Texture2D textures[] + NonUniformResourceIndex).
    // Explicit Init/Destroy pair with a leak-trap destructor (RHI discipline).
    class BindlessSet {
    public:
        static constexpr uint32_t kMaxImages = 8192;
        static constexpr uint32_t kMaxSamplers = 32;

        BindlessSet();
        ~BindlessSet();

        BindlessSet(const BindlessSet&) = delete;
        BindlessSet& operator=(const BindlessSet&) = delete;

        // Creates the pool/layout/set and registers the default samplers
        // (nearest @0, linear @1), like the old engine did.
        bool Init(Device& device);

        // Idempotent; the destructor aborts if initialized but not destroyed.
        void Destroy();

        // Writes the image/sampler at the given array index.
        bool AddImage(uint32_t id, const Image& image);
        bool AddSampler(uint32_t id, const Sampler& sampler);

        uint32_t GetImageCapacity() const;
        uint32_t GetSamplerCapacity() const;

        bool IsValid() const;

        // Binds this set to the pipeline's set slot. The pipeline's shader must
        // declare matching bindings (unbounded sampled-image array @0, sampler
        // array @1).
        void Bind(CommandList& cmd, const GraphicsPipeline& pipeline, uint32_t setIndex = 0);
        void Bind(CommandList& cmd, const ComputePipeline& pipeline, uint32_t setIndex = 0);

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::rhi
