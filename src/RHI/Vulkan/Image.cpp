#include "RHI/Image.hpp"
#include <Core/Profile.hpp>

#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <utility>

namespace moe::rhi {
    Image::Image() = default;

    Image::~Image() {
        MOE_RHI_ASSERT(mImpl == nullptr || !mImpl->mOwned,
                "Image leaked: Destroy() not called");
    }

    void Image::Destroy() {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mOwned && mImpl->mDevice && mImpl->mImage != VK_NULL_HANDLE) {
            // Deferred: the image may still be in flight when destroyed.
            DeferredDeletion deletion;
            deletion.mImage = mImpl->mImage;
            deletion.mImageView = mImpl->mView;
            deletion.mAllocation = mImpl->mAllocation;
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
        }
        mImpl.reset();
    }

    ImageType Image::GetType() const {
        return mImpl ? mImpl->mType : ImageType::k2D;
    }

    uint32_t Image::GetWidth() const {
        return mImpl ? mImpl->mWidth : 0;
    }

    uint32_t Image::GetHeight() const {
        return mImpl ? mImpl->mHeight : 0;
    }

    uint32_t Image::GetDepth() const {
        return mImpl ? mImpl->mDepth : 0;
    }

    uint32_t Image::GetMipLevels() const {
        return mImpl ? mImpl->mMipLevels : 0;
    }

    uint32_t Image::GetLayerCount() const {
        return mImpl ? mImpl->mLayerCount : 0;
    }

    Format Image::GetFormat() const {
        return mImpl ? mImpl->mFormat : Format::kUndefined;
    }

    ImageUsage Image::GetUsage() const {
        return mImpl ? mImpl->mUsage : ImageUsage::kSampled;
    }
}// namespace moe::rhi