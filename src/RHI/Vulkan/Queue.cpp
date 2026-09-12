#include "RHI/Queue.hpp"
#include <Core/Profile.hpp>

#include "RHI/CommandList.hpp"
#include "RHI/Fence.hpp"
#include "RHI/TimelineSemaphore.hpp"
#include "Mappings.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <vector>

namespace moe::rhi {
    bool Queue::Submit(const CommandList& commandList, const SubmitInfo& submitInfo, Fence* fence) {
        MOE_PROFILE_ZONE();
        if (mQueue == 0 || commandList.mImpl == nullptr) {
            return false;
        }

        std::vector<VkSemaphoreSubmitInfo> waits;
        waits.reserve(submitInfo.mWaits.size());
        for (const auto& wait : submitInfo.mWaits) {
            VkSemaphoreSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            info.semaphore = wait.mSemaphore->mImpl->mSemaphore;
            info.value = wait.mValue;
            info.stageMask = ToVkPipelineStage(wait.mStage);
            waits.push_back(info);
        }
        std::vector<VkSemaphoreSubmitInfo> signals;
        signals.reserve(submitInfo.mSignals.size());
        for (const auto& signal : submitInfo.mSignals) {
            VkSemaphoreSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            info.semaphore = signal.mSemaphore->mImpl->mSemaphore;
            info.value = signal.mValue;
            // Signal once every command in the submission has completed.
            info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            signals.push_back(info);
        }

        VkCommandBufferSubmitInfo commandInfo{};
        commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandInfo.commandBuffer = commandList.mImpl->mCommandBuffer;

        VkSubmitInfo2 info{};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        info.waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size());
        info.pWaitSemaphoreInfos = waits.data();
        info.commandBufferInfoCount = 1;
        info.pCommandBufferInfos = &commandInfo;
        info.signalSemaphoreInfoCount = static_cast<uint32_t>(signals.size());
        info.pSignalSemaphoreInfos = signals.data();

        const VkFence vkFence = fence != nullptr && fence->mImpl != nullptr
                ? fence->mImpl->mFence
                : VK_NULL_HANDLE;
        return vkQueueSubmit2(reinterpret_cast<VkQueue>(mQueue), 1, &info, vkFence) == VK_SUCCESS;
    }

    bool Queue::Submit(const CommandList& commandList, bool waitForCompletion) {
        MOE_PROFILE_ZONE();
        if (mQueue == 0 || commandList.mImpl == nullptr) {
            return false;
        }
        const VkQueue queue = reinterpret_cast<VkQueue>(mQueue);
        VkCommandBufferSubmitInfo commandInfo{};
        commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandInfo.commandBuffer = commandList.mImpl->mCommandBuffer;

        VkSubmitInfo2 info{};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        info.commandBufferInfoCount = 1;
        info.pCommandBufferInfos = &commandInfo;

        if (!waitForCompletion) {
            return vkQueueSubmit2(queue, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS;
        }

        const VkDevice device = reinterpret_cast<VkDevice>(mDevice);
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            return false;
        }
        if (vkQueueSubmit2(queue, 1, &info, fence) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            return false;
        }
        const bool ok = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
        vkDestroyFence(device, fence, nullptr);
        return ok;
    }

    bool Queue::WaitIdle() {
        MOE_PROFILE_ZONE();
        if (mQueue == 0) {
            return false;
        }
        return vkQueueWaitIdle(reinterpret_cast<VkQueue>(mQueue)) == VK_SUCCESS;
    }
}// namespace moe::rhi
