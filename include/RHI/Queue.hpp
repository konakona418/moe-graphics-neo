#pragma once

#include "RHI/RHICommon.hpp"

#include <cstdint>

namespace moe::rhi {
    class Device;
    class CommandList;
    class Fence;

    // Non-owning handle to a device queue (same lifecycle class as
    // GraphicsPipeline: obtained from Device::GetQueue, no Create/Destroy).
    // The queue itself is owned by the Device.
    class Queue {
    public:
        Queue() = default;
        ~Queue() = default;

        Queue(const Queue&) = delete;
        Queue& operator=(const Queue&) = delete;

        bool IsValid() const {
            return mQueue != 0;
        }

        QueueType GetType() const {
            return mType;
        }

        uint32_t GetFamily() const {
            return mFamily;
        }

        // Submits a command list to this queue (same SubmitInfo semantics as
        // Device::Submit). The optional fence is signaled on completion.
        bool Submit(const CommandList& commandList, const SubmitInfo& submitInfo,
                Fence* fence = nullptr);

        // Submits without timeline waits/signals. When waitForCompletion is
        // true the host blocks on a temporary fence.
        bool Submit(const CommandList& commandList, bool waitForCompletion);

        bool WaitIdle();

        // Opaque VkQueue handle (see Device::GetVulkanHandles).
        uintptr_t GetVulkanHandle() const {
            return mQueue;
        }

    private:
        friend class Device;

        uintptr_t mQueue{0};
        uintptr_t mDevice{0}; // opaque VkDevice, for temporary fences
        uint32_t mFamily{0};
        QueueType mType{QueueType::kGraphics};
    };
}// namespace moe::rhi
