#include "RHI/Sampler.hpp"

#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <utility>

namespace moe::rhi {
    Sampler::Sampler() = default;

    Sampler::~Sampler() {
        MOE_RHI_ASSERT(mImpl == nullptr, "Sampler leaked: Destroy() not called");
    }

    void Sampler::Destroy() {
        if (mImpl && mImpl->mDevice && mImpl->mSampler != VK_NULL_HANDLE) {
            // Deferred: the sampler may still be in flight when destroyed.
            DeferredDeletion deletion;
            deletion.mSampler = mImpl->mSampler;
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
        }
        mImpl.reset();
    }
}// namespace moe::rhi
