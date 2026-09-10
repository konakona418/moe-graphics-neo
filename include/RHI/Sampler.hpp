#pragma once

#include "RHI/RHICommon.hpp"

#include <memory>

namespace moe::rhi {
    class Device;
    class DescriptorSet;
    struct SamplerImpl;

    // GPU sampler. Created via Device::CreateSampler; destroyed explicitly via
    // Destroy() (the destructor is a leak trap). Used to sample Image resources
    // in shaders (bound separately from images via DescriptorSet::WriteSampler).
    class Sampler {
    public:
        Sampler();
        ~Sampler();

        Sampler(const Sampler&) = delete;
        Sampler& operator=(const Sampler&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the sampler
        // was created but not destroyed (leak trap).
        void Destroy();

    private:
        friend class Device;
        friend class DescriptorSet;
        friend class BindlessSet;

        std::unique_ptr<SamplerImpl> mImpl;
    };
}// namespace moe::rhi
