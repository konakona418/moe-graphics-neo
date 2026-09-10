#pragma once

#include "RHI/RHICommon.hpp"

#include <memory>
#include <string>

namespace moe::rhi {
    class Buffer;
    class Image;
    class Sampler;
    class CommandList;
    class Swapchain;
    class PipelineCache;
    class DescriptorSetLayout;
    class DescriptorSet;
    class GraphicsPipeline;
    class ComputePipeline;
    struct GraphicsPipelineState;
    struct ComputePipelineState;
    struct DeviceImpl;

    struct DeviceCreateInfo {
        bool mEnableValidation{false};
        // Enables Vulkan surface extensions so a swapchain can be created from
        // a caller-supplied surface. Headless usage leaves this false.
        bool mEnablePresent{false};
        // Requests the Vulkan 1.2 descriptor-indexing feature set (bindless
        // descriptor arrays; see BindlessSet). Off by default.
        bool mEnableDescriptorIndexing{false};
        std::string mApplicationName{"moe-rhi"};
        PipelineCache* mPipelineCache{nullptr}; // required; null is a fatal error
    };

    // Opaque Vulkan handles for integrations that must talk to Vulkan directly
    // (e.g. ImGui). Not for normal RHI usage; the RHI keeps its abstraction
    // and makes no guarantees about these beyond "valid while the device is".
    struct RhiVulkanHandles {
        uintptr_t mInstance{0};
        uintptr_t mPhysicalDevice{0};
        uintptr_t mDevice{0};
        uintptr_t mGraphicsQueue{0};
        uint32_t mGraphicsQueueFamily{0};
    };

    class Device {
    public:
        Device();

        // Creates the device. Returns false on failure; moe::Error::Get()
        // then holds the reason. The device must outlive all resources it
        // created.
        static bool Create(const DeviceCreateInfo& info, Device& outDevice);

        // Explicit teardown (idempotent). Must be called before the object
        // goes out of scope; the destructor aborts if the device is still
        // alive (leak trap). Destroy all resources created from this device,
        // and the injected PipelineCache, before calling this.
        void Destroy();

        ~Device();

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        bool CreateBuffer(const BufferCreateInfo& info, Buffer& outBuffer);
        bool CreateImage(const ImageCreateInfo& info, Image& outImage);
        bool CreateSampler(const SamplerCreateInfo& info, Sampler& outSampler);
        bool CreateCommandList(CommandList& outCommandList);
        // Creates a present-capable swapchain from an opaque VkSurfaceKHR handle
        // (created by the caller, e.g. neo::Window). The graphics queue is used
        // for present; the surface must be compatible with it.
        bool CreateSwapchain(uintptr_t surfaceHandle, uint32_t width, uint32_t height, Swapchain& outSwapchain);

        // Descriptor set allocated from the device's internal pool. The layout
        // must come from a pipeline's GetDescriptorSetLayout.
        bool CreateDescriptorSet(const DescriptorSetLayout& layout, DescriptorSet& outSet);

        // Pipeline creation is routed through the injected PipelineCache.
        bool GetOrCreateGraphicsPipeline(const GraphicsPipelineState& state, GraphicsPipeline& out);
        bool GetOrCreateComputePipeline(const ComputePipelineState& state, ComputePipeline& out);

        // Submits the recorded command list to the graphics queue. When
        // waitForCompletion is true, blocks until the GPU finishes.
        bool Submit(const CommandList& commandList, bool waitForCompletion);

        bool WaitIdle();

        // Returns the underlying VkInstance as an opaque handle (used to create
        // a surface via GLFW; requires mEnablePresent).
        bool GetInstanceHandle(uintptr_t& outInstance) const;

        // Returns opaque Vulkan handles for integrations that must talk to
        // Vulkan directly (e.g. ImGui). See RhiVulkanHandles.
        bool GetVulkanHandles(RhiVulkanHandles& outHandles) const;

    private:
        friend class DefaultPipelineCache;
        friend class BindlessSet;

        PipelineCache* mPipelineCache{nullptr};
        std::unique_ptr<DeviceImpl> mImpl;
    };
}// namespace moe::rhi