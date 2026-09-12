#include "RHI/TimelineSemaphore.hpp"
#include <Core/Profile.hpp>

#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <utility>

namespace moe::rhi {
    TimelineSemaphore::TimelineSemaphore() = default;

    TimelineSemaphore::~TimelineSemaphore() {
        MOE_RHI_ASSERT(mImpl == nullptr, "TimelineSemaphore leaked: Destroy() not called");
    }

    void TimelineSemaphore::Destroy() {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mDevice && mImpl->mSemaphore != VK_NULL_HANDLE) {
            // Deferred: the semaphore may still be in flight when destroyed.
            DeferredDeletion deletion;
            deletion.mSemaphore = mImpl->mSemaphore;
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
        }
        mImpl.reset();
    }

    uint64_t TimelineSemaphore::GetValue() const {
        if (mImpl == nullptr || mImpl->mSemaphore == VK_NULL_HANDLE) {
            return 0;
        }
        uint64_t value = 0;
        vkGetSemaphoreCounterValue(mImpl->mDevice->mDevice, mImpl->mSemaphore, &value);
        return value;
    }

    bool TimelineSemaphore::IsReached(uint64_t value) const {
        return GetValue() >= value;
    }

    bool TimelineSemaphore::Wait(uint64_t value, uint64_t timeoutNs) const {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || mImpl->mSemaphore == VK_NULL_HANDLE) {
            return false;
        }
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &mImpl->mSemaphore;
        waitInfo.pValues = &value;
        return vkWaitSemaphores(mImpl->mDevice->mDevice, &waitInfo, timeoutNs) == VK_SUCCESS;
    }

    bool TimelineSemaphore::Signal(uint64_t value) {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || mImpl->mSemaphore == VK_NULL_HANDLE) {
            return false;
        }
        VkSemaphoreSignalInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signalInfo.semaphore = mImpl->mSemaphore;
        signalInfo.value = value;
        return vkSignalSemaphore(mImpl->mDevice->mDevice, &signalInfo) == VK_SUCCESS;
    }
}// namespace moe::rhi
