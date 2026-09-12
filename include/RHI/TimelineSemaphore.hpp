#pragma once

#include "RHI/RHICommon.hpp"

#include <memory>

namespace moe::rhi {
    class Device;
    struct TimelineSemaphoreImpl;

    // Timeline semaphore: a monotonically increasing 64-bit counter used to
    // track GPU completion. Values are signaled by queue submissions (see
    // SubmitInfo) or from the host (Signal), and waited on without blocking
    // the submitting thread (Wait/IsReached). Created via
    // Device::CreateTimelineSemaphore; destroyed explicitly via Destroy() (the
    // destructor is a leak trap).
    class TimelineSemaphore {
    public:
        TimelineSemaphore();
        ~TimelineSemaphore();

        TimelineSemaphore(const TimelineSemaphore&) = delete;
        TimelineSemaphore& operator=(const TimelineSemaphore&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the
        // semaphore was created but not destroyed (leak trap).
        void Destroy();

        // Current counter value (0 for an invalid semaphore).
        uint64_t GetValue() const;

        // True once the counter has reached `value`.
        bool IsReached(uint64_t value) const;

        // Blocks until the counter reaches `value` or the timeout elapses.
        // Returns true if reached, false on timeout or error.
        bool Wait(uint64_t value, uint64_t timeoutNs = UINT64_MAX) const;

        // Host-side signal: the counter must be strictly greater than the
        // current value (Vulkan requirement). Returns false on failure.
        bool Signal(uint64_t value);

    private:
        friend class Device;

        std::unique_ptr<TimelineSemaphoreImpl> mImpl;
    };
}// namespace moe::rhi
