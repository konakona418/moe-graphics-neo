#include "RHI/PipelineState.hpp"

#include "RHI/Shader.hpp"

namespace moe::rhi {
    namespace {
        uint64_t Combine(uint64_t hash, uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
            return hash;
        }

        template<typename T>
        uint64_t HashScalar(T value) {
            return Combine(1469598103934665603ull, static_cast<uint64_t>(value));
        }

        template<typename T, size_t N>
        uint64_t HashArray(const std::array<T, N>& values, uint32_t count) {
            uint64_t hash = 1469598103934665603ull;
            for (uint32_t i = 0; i < count; ++i) {
                hash = Combine(hash, HashScalar(values[i]));
            }
            return hash;
        }
    }// namespace

    uint64_t GraphicsPipelineState::GetHash() const {
        uint64_t hash = 1469598103934665603ull;
        hash = Combine(hash, mProgram ? mProgram->GetContentHash() : 0);
        hash = Combine(hash, static_cast<uint64_t>(mTopology));
        hash = Combine(hash, static_cast<uint64_t>(mRaster.mPolygonMode));
        hash = Combine(hash, static_cast<uint64_t>(mRaster.mCullMode));
        hash = Combine(hash, static_cast<uint64_t>(mRaster.mFrontFace));
        hash = Combine(hash, static_cast<uint64_t>(mRaster.mDepthClampEnable));
        hash = Combine(hash, static_cast<uint64_t>(mRaster.mDepthBiasEnable));
        hash = Combine(hash, static_cast<uint64_t>(mDepth.mTestEnable));
        hash = Combine(hash, static_cast<uint64_t>(mDepth.mWriteEnable));
        hash = Combine(hash, static_cast<uint64_t>(mDepth.mCompareOp));
        hash = Combine(hash, static_cast<uint64_t>(mStencil.mFailOp));
        hash = Combine(hash, static_cast<uint64_t>(mStencil.mPassOp));
        hash = Combine(hash, static_cast<uint64_t>(mStencil.mDepthFailOp));
        hash = Combine(hash, static_cast<uint64_t>(mStencil.mCompareOp));
        hash = Combine(hash, mStencil.mCompareMask);
        hash = Combine(hash, mStencil.mWriteMask);
        hash = Combine(hash, static_cast<uint64_t>(mMultisample.mSampleShading));
        hash = Combine(hash, mMultisample.mSampleCount);
        for (uint32_t i = 0; i < mBlendAttachmentCount; ++i) {
            const auto& b = mBlendAttachments[i];
            hash = Combine(hash, static_cast<uint64_t>(b.mBlendEnabled));
            hash = Combine(hash, static_cast<uint64_t>(b.mSrcColor));
            hash = Combine(hash, static_cast<uint64_t>(b.mDstColor));
            hash = Combine(hash, static_cast<uint64_t>(b.mColorOp));
            hash = Combine(hash, static_cast<uint64_t>(b.mSrcAlpha));
            hash = Combine(hash, static_cast<uint64_t>(b.mDstAlpha));
            hash = Combine(hash, static_cast<uint64_t>(b.mAlphaOp));
        }
        hash = Combine(hash, mBlendAttachmentCount);
        hash = Combine(hash, HashArray(mColorFormats, mColorFormatCount));
        hash = Combine(hash, mColorFormatCount);
        hash = Combine(hash, static_cast<uint64_t>(mDepthFormat));
        for (uint32_t i = 0; i < mVertexAttributeCount; ++i) {
            const auto& a = mVertexAttributes[i];
            hash = Combine(hash, a.mLocation);
            hash = Combine(hash, a.mBinding);
            hash = Combine(hash, static_cast<uint64_t>(a.mFormat));
            hash = Combine(hash, a.mOffset);
        }
        hash = Combine(hash, mVertexAttributeCount);
        for (uint32_t i = 0; i < mVertexBindingCount; ++i) {
            const auto& b = mVertexBindings[i];
            hash = Combine(hash, b.mBinding);
            hash = Combine(hash, b.mStride);
            hash = Combine(hash, static_cast<uint64_t>(b.mPerInstance));
        }
        hash = Combine(hash, mVertexBindingCount);
        hash = Combine(hash, mSpecializationHash);
        return hash;
    }

    bool GraphicsPipelineState::operator==(const GraphicsPipelineState& other) const {
        return mProgram == other.mProgram
                && mTopology == other.mTopology
                && mRaster.mPolygonMode == other.mRaster.mPolygonMode
                && mRaster.mCullMode == other.mRaster.mCullMode
                && mRaster.mFrontFace == other.mRaster.mFrontFace
                && mRaster.mDepthClampEnable == other.mRaster.mDepthClampEnable
                && mRaster.mDepthBiasEnable == other.mRaster.mDepthBiasEnable
                && mDepth.mTestEnable == other.mDepth.mTestEnable
                && mDepth.mWriteEnable == other.mDepth.mWriteEnable
                && mDepth.mCompareOp == other.mDepth.mCompareOp
                && mStencil.mFailOp == other.mStencil.mFailOp
                && mStencil.mPassOp == other.mStencil.mPassOp
                && mStencil.mDepthFailOp == other.mStencil.mDepthFailOp
                && mStencil.mCompareOp == other.mStencil.mCompareOp
                && mStencil.mCompareMask == other.mStencil.mCompareMask
                && mStencil.mWriteMask == other.mStencil.mWriteMask
                && mMultisample.mSampleShading == other.mMultisample.mSampleShading
                && mMultisample.mSampleCount == other.mMultisample.mSampleCount
                && mBlendAttachments == other.mBlendAttachments
                && mBlendAttachmentCount == other.mBlendAttachmentCount
                && mColorFormats == other.mColorFormats
                && mColorFormatCount == other.mColorFormatCount
                && mDepthFormat == other.mDepthFormat
                && mVertexAttributes == other.mVertexAttributes
                && mVertexAttributeCount == other.mVertexAttributeCount
                && mVertexBindings == other.mVertexBindings
                && mVertexBindingCount == other.mVertexBindingCount
                && mSpecializationHash == other.mSpecializationHash;
    }

    uint64_t ComputePipelineState::GetHash() const {
        uint64_t hash = 1469598103934665603ull;
        hash = Combine(hash, mProgram ? mProgram->GetContentHash() : 0);
        hash = Combine(hash, mSpecializationHash);
        return hash;
    }

    bool ComputePipelineState::operator==(const ComputePipelineState& other) const {
        return mProgram == other.mProgram && mSpecializationHash == other.mSpecializationHash;
    }
}// namespace moe::rhi