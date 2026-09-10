#pragma once

#include "RHI/RHICommon.hpp"

#include <cstdint>
#include <memory>

namespace moe::rhi {
    class Device;
    class CommandList;
    class ComputePipeline;
    struct BufferImpl;

    class Buffer {
    public:
        Buffer();
        ~Buffer();

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the buffer
        // was created but not destroyed (leak trap).
        void Destroy();

        // Returns a pointer to the host-visible memory, or nullptr if the
        // buffer was not created cpu-visible.
        void* Map();
        void Unmap();
        uint64_t GetDeviceAddress() const;
        uint64_t GetSize() const;

    private:
        friend class Device;
        friend class CommandList;
        friend class DescriptorSet;

        std::unique_ptr<BufferImpl> mImpl;
    };
}// namespace moe::rhi