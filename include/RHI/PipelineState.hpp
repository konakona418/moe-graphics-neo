#pragma once

#include "RHI/RHICommon.hpp"

#include <array>
#include <compare>

namespace moe::rhi {
    class ShaderProgram;

    constexpr uint32_t kMaxColorAttachments = 8;
    constexpr uint32_t kMaxVertexAttributes = 16;
    constexpr uint32_t kMaxVertexBindings = 4;

    struct ColorBlendAttachmentState {
        bool mBlendEnabled{false};
        BlendFactor mSrcColor{BlendFactor::kOne};
        BlendFactor mDstColor{BlendFactor::kZero};
        BlendOp mColorOp{BlendOp::kAdd};
        BlendFactor mSrcAlpha{BlendFactor::kOne};
        BlendFactor mDstAlpha{BlendFactor::kZero};
        BlendOp mAlphaOp{BlendOp::kAdd};

        auto operator<=>(const ColorBlendAttachmentState&) const = default;
    };

    struct RasterState {
        PolygonMode mPolygonMode{PolygonMode::kFill};
        CullMode mCullMode{CullMode::kBack};
        FrontFace mFrontFace{FrontFace::kCounterClockwise};
        bool mDepthClampEnable{false};
        bool mDepthBiasEnable{false};

        auto operator<=>(const RasterState&) const = default;
    };

    struct DepthState {
        bool mTestEnable{false};
        bool mWriteEnable{true};
        CompareOp mCompareOp{CompareOp::kLess};

        auto operator<=>(const DepthState&) const = default;
    };

    struct StencilOpState {
        StencilOp mFailOp{StencilOp::kKeep};
        StencilOp mPassOp{StencilOp::kKeep};
        StencilOp mDepthFailOp{StencilOp::kKeep};
        CompareOp mCompareOp{CompareOp::kAlways};
        uint32_t mCompareMask{0xff};
        uint32_t mWriteMask{0xff};

        auto operator<=>(const StencilOpState&) const = default;
    };

    struct MultisampleState {
        bool mSampleShading{false};
        uint8_t mSampleCount{1}; // 1, 2, 4, 8

        auto operator<=>(const MultisampleState&) const = default;
    };

    struct VertexAttribute {
        uint32_t mLocation{0};
        uint32_t mBinding{0};
        Format mFormat{Format::kR32G32B32A32Float};
        uint32_t mOffset{0};

        auto operator<=>(const VertexAttribute&) const = default;
    };

    struct VertexBinding {
        uint32_t mBinding{0};
        uint32_t mStride{0};
        bool mPerInstance{false};

        auto operator<=>(const VertexBinding&) const = default;
    };

    struct GraphicsPipelineState {
        const ShaderProgram* mProgram{nullptr};
        PrimitiveTopology mTopology{PrimitiveTopology::kTriangleList};

        RasterState mRaster;
        DepthState mDepth;
        StencilOpState mStencil;

        MultisampleState mMultisample;

        std::array<ColorBlendAttachmentState, kMaxColorAttachments> mBlendAttachments{};
        uint32_t mBlendAttachmentCount{0};

        std::array<Format, kMaxColorAttachments> mColorFormats{};
        uint32_t mColorFormatCount{0};
        Format mDepthFormat{Format::kUndefined};

        std::array<VertexAttribute, kMaxVertexAttributes> mVertexAttributes{};
        uint32_t mVertexAttributeCount{0};
        std::array<VertexBinding, kMaxVertexBindings> mVertexBindings{};
        uint32_t mVertexBindingCount{0};

        // Placeholder for specialization constants; 0 = none.
        uint32_t mSpecializationHash{0};

        // Includes the shader program's content hash, so a reload produces a
        // different key.
        uint64_t GetHash() const;

        bool operator==(const GraphicsPipelineState&) const;
    };

    struct ComputePipelineState {
        const ShaderProgram* mProgram{nullptr};
        uint32_t mSpecializationHash{0};

        uint64_t GetHash() const;

        bool operator==(const ComputePipelineState&) const;
    };
}// namespace moe::rhi