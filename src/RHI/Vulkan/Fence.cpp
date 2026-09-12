#include "RHI/Fence.hpp"
#include <Core/Profile.hpp>

#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <utility>

namespace moe::rhi {
    Fence::Fence() = default;

    Fence::~Fence() {
        MOE_RHI_ASSERT(mImpl == nullptr, "Fence leaked: Destroy() not called");
    }

    void Fence::Destroy() {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mDevice && mImpl->mFence != VK_NULL_HANDLE) {
            // Deferred: the fence may still be in flight when destroyed.
            DeferredDeletion deletion;
            deletion.mFence = mImpl->mFence;
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
        }
        mImpl.reset();
    }

    void Fence::Reset() {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mFence != VK_NULL_HANDLE) {
            vkResetFences(mImpl->mDevice->mDevice, 1, &mImpl->mFence);
        }
    }

    bool Fence::IsSignaled() const {
        if (mImpl == nullptr || mImpl->mFence == VK_NULL_HANDLE) {
            return false;
        }
        return vkGetFenceStatus(mImpl->mDevice->mDevice, mImpl->mFence) == VK_SUCCESS;
    }

    bool Fence::Wait(uint64_t timeoutNs) const {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || mImpl->mFence == VK_NULL_HANDLE) {
            return false;
        }
        return vkWaitForFences(mImpl->mDevice->mDevice, 1, &mImpl->mFence, VK_TRUE, timeoutNs)
                == VK_SUCCESS;
    }
}// namespace moe::rhi
