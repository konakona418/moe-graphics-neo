#include "RHI/Buffer.hpp"

#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <utility>

namespace moe::rhi {
    Buffer::Buffer() = default;

    Buffer::~Buffer() {
        MOE_RHI_ASSERT(mImpl == nullptr, "Buffer leaked: Destroy() not called");
    }

    void Buffer::Destroy() {
        if (mImpl && mImpl->mDevice && mImpl->mBuffer != VK_NULL_HANDLE) {
            // Deferred: the buffer may still be in flight when destroyed.
            DeferredDeletion deletion;
            deletion.mBuffer = mImpl->mBuffer;
            deletion.mAllocation = mImpl->mAllocation;
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
        }
        mImpl.reset();
    }

    void* Buffer::Map() {
        if (!mImpl || !mImpl->mCpuVisible) {
            return nullptr;
        }
        void* mapped = nullptr;
        if (vmaMapMemory(mImpl->mDevice->mAllocator, mImpl->mAllocation, &mapped) != VK_SUCCESS) {
            return nullptr;
        }
        // Refresh data written by the GPU (no-op on coherent memory).
        vmaInvalidateAllocation(mImpl->mDevice->mAllocator, mImpl->mAllocation, 0, mImpl->mSize);
        return mapped;
    }

    void Buffer::Unmap() {
        if (mImpl && mImpl->mCpuVisible) {
            // Push data written by the CPU (no-op on coherent memory).
            vmaFlushAllocation(mImpl->mDevice->mAllocator, mImpl->mAllocation, 0, mImpl->mSize);
            vmaUnmapMemory(mImpl->mDevice->mAllocator, mImpl->mAllocation);
        }
    }

    uint64_t Buffer::GetDeviceAddress() const {
        return mImpl ? mImpl->mDeviceAddress : 0;
    }

    uint64_t Buffer::GetSize() const {
        return mImpl ? mImpl->mSize : 0;
    }
}// namespace moe::rhi