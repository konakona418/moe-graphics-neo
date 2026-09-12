#pragma once

#include "RHI/RHICommon.hpp"

#include <memory>

namespace moe::rhi {
    class Device;
    struct FenceImpl;

    // Binary GPU fence. Created via Device::CreateFence; destroyed explicitly
    // via Destroy() (the destructor is a leak trap). A fence can be attached
    // to a submission (Device::Submit) and polled/waited without blocking the
    // submitting thread.
    class Fence {
    public:
        Fence();
        ~Fence();

        Fence(const Fence&) = delete;
        Fence& operator=(const Fence&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the fence
        // was created but not destroyed (leak trap).
        void Destroy();

        // Returns the fence to the unsignaled state.
        void Reset();

        // True if the fence is currently signaled.
        bool IsSignaled() const;

        // Blocks until the fence is signaled or the timeout elapses. Returns
        // true if signaled, false on timeout or error.
        bool Wait(uint64_t timeoutNs = UINT64_MAX) const;

    private:
        friend class Device;
        friend class Queue;

        std::unique_ptr<FenceImpl> mImpl;
    };
}// namespace moe::rhi
