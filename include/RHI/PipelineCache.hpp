#pragma once

#include "RHI/PipelineState.hpp"

#include <memory>

namespace moe::rhi {
    class Device;
    class ShaderProgram;
    class GraphicsPipeline;
    class ComputePipeline;

    // Injectable pipeline cache. Device::Create requires one (a null cache is
    // a fatal error); pipeline creation is routed through whichever
    // implementation the user injected. Implementations log their name so the
    // active one is visible.
    class PipelineCache {
    public:
        virtual ~PipelineCache() = default;

        // Called by Device::Create once the device exists. The cache must be
        // destroyed before the device.
        virtual bool Create(Device& device) = 0;

        virtual const char* GetName() const = 0;

        virtual bool GetOrCreateGraphics(const GraphicsPipelineState& state, GraphicsPipeline& out) = 0;
        virtual bool GetOrCreateCompute(const ComputePipelineState& state, ComputePipeline& out) = 0;

        // Re-reads every stage of the program's shaders, re-reflects, and
        // invalidates cached pipelines built from it.
        virtual bool Reload(ShaderProgram& program) = 0;

        // Destroys all cached pipelines (through the device's deferred
        // deletion queue).
        virtual void Clear() = 0;
    };

    // The stock cache: in-process hash dedup + a driver-level VkPipelineCache
    // (disk persistence) so identical pipeline states compile once.
    class DefaultPipelineCache : public PipelineCache {
    public:
        DefaultPipelineCache();
        ~DefaultPipelineCache() override;

        DefaultPipelineCache(const DefaultPipelineCache&) = delete;
        DefaultPipelineCache& operator=(const DefaultPipelineCache&) = delete;

        // Binds the cache to a device (called by Device::Create). Explicit
        // teardown via Destroy(); the destructor aborts if the cache was
        // created but not destroyed (leak trap). Destroy before the device.
        bool Create(Device& device) override;
        void Destroy();

        const char* GetName() const override;
        bool GetOrCreateGraphics(const GraphicsPipelineState& state, GraphicsPipeline& out) override;
        bool GetOrCreateCompute(const ComputePipelineState& state, ComputePipeline& out) override;
        bool Reload(ShaderProgram& program) override;
        void Clear() override;

        // Number of distinct cached pipelines (diagnostics / tests).
        uint32_t GetNodeCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::rhi