#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe::rhi {
    enum class BufferUsage : uint32_t {
        kUniform = 1 << 0,
        kStorage = 1 << 1,
        kVertex = 1 << 2,
        kIndex = 1 << 3,
        kTransferSrc = 1 << 4,
        kTransferDst = 1 << 5,
    };

    constexpr BufferUsage operator|(BufferUsage lhs, BufferUsage rhs) {
        return static_cast<BufferUsage>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
    }

    constexpr BufferUsage operator&(BufferUsage lhs, BufferUsage rhs) {
        return static_cast<BufferUsage>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
    }

    constexpr BufferUsage& operator|=(BufferUsage& lhs, BufferUsage rhs) {
        lhs = lhs | rhs;
        return lhs;
    }

    constexpr bool HasFlag(BufferUsage value, BufferUsage flag) {
        return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
    }

    enum class Format : uint32_t {
        kUndefined,
        kR8G8B8A8Unorm,
        kR8G8B8A8Srgb,
        kB8G8R8A8Unorm,
        kB8G8R8A8Srgb,
        kR16G16Float,
        kR32Float,
        kR32Uint,
        kR32G32Float,
        kR32G32B32Float,
        kR16G16B16A16Float,
        kR32G32B32A32Float,
        kD32Float,
    };

    enum class ShaderStage : uint32_t {
        kVertex,
        kFragment,
        kGeometry,
        kCompute,
    };

    enum class DescriptorType : uint32_t {
        kUniformBuffer,
        kStorageBuffer,
        kCombinedImageSampler,
        kSampledImage,
        kStorageImage,
        kSampler,
    };

    enum class ImageType : uint32_t {
        k2D,
        k3D,
        kCube,
    };

    enum class ImageUsage : uint32_t {
        kSampled = 1 << 0,
        kStorage = 1 << 1,
        kColorAttachment = 1 << 2,
        kDepthAttachment = 1 << 3,
        kTransferSrc = 1 << 4,
        kTransferDst = 1 << 5,
    };

    constexpr ImageUsage operator|(ImageUsage lhs, ImageUsage rhs) {
        return static_cast<ImageUsage>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
    }

    constexpr bool HasFlag(ImageUsage value, ImageUsage flag) {
        return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
    }

    enum class ImageLayout : uint32_t {
        kUndefined,
        kGeneral,
        kColorAttachment,
        kDepthStencilAttachment,
        kShaderReadOnly,
        kTransferSrc,
        kTransferDst,
        kPresentSrc,
    };

    enum class Filter : uint32_t {
        kNearest,
        kLinear,
    };

    enum class LoadOp : uint32_t {
        kClear,
        kLoad, // keep the previous contents (e.g. compositing on top of a scene)
    };

    enum class AddressMode : uint32_t {
        kRepeat,
        kMirroredRepeat,
        kClampToEdge,
        kClampToBorder,
    };

    enum class PrimitiveTopology : uint32_t {
        kPointList,
        kLineList,
        kTriangleList,
        kTriangleStrip,
    };

    enum class PolygonMode : uint32_t {
        kFill,
        kLine,
        kPoint,
    };

    enum class CullMode : uint32_t {
        kNone,
        kFront,
        kBack,
    };

    enum class FrontFace : uint32_t {
        kCounterClockwise,
        kClockwise,
    };

    enum class BlendFactor : uint32_t {
        kZero,
        kOne,
        kSrcColor,
        kOneMinusSrcColor,
        kDstColor,
        kOneMinusDstColor,
        kSrcAlpha,
        kOneMinusSrcAlpha,
        kDstAlpha,
        kOneMinusDstAlpha,
        kSrcAlphaSaturate,
    };

    enum class BlendOp : uint32_t {
        kAdd,
        kSubtract,
        kReverseSubtract,
        kMin,
        kMax,
    };

    enum class CompareOp : uint32_t {
        kNever,
        kLess,
        kEqual,
        kLessEqual,
        kGreater,
        kNotEqual,
        kGreaterEqual,
        kAlways,
    };

    enum class StencilOp : uint32_t {
        kKeep,
        kZero,
        kReplace,
        kIncrementClamp,
        kDecrementClamp,
        kInvert,
        kIncrementWrap,
        kDecrementWrap,
    };

    enum class PipelineStage : uint32_t {
        kTopOfPipe,
        kDrawIndirect,
        kVertexInput,
        kVertexShader,
        kGeometryShader,
        kFragmentShader,
        kEarlyFragmentTests,
        kLateFragmentTests,
        kColorAttachmentOutput,
        kComputeShader,
        kTransfer,
        kHost,
        kBottomOfPipe,
    };

    enum class Access : uint32_t {
        kNone,
        kIndirectCommandRead,
        kIndexRead,
        kVertexAttributeRead,
        kUniformRead,
        kInputAttachmentRead,
        kShaderRead,
        kShaderWrite,
        kColorAttachmentRead,
        kColorAttachmentWrite,
        kDepthStencilAttachmentRead,
        kDepthStencilAttachmentWrite,
        kTransferRead,
        kTransferWrite,
        kHostRead,
        kHostWrite,
        kMemoryRead,
        kMemoryWrite,
    };

    // Describes one memory dependency: waits until the source stage's accesses
    // complete, then makes them visible to the destination stage.
    struct SyncInfo {
        PipelineStage mSrcStage{PipelineStage::kBottomOfPipe};
        Access mSrcAccess{Access::kNone};
        PipelineStage mDstStage{PipelineStage::kTopOfPipe};
        Access mDstAccess{Access::kNone};
    };

    struct BufferCreateInfo {
        uint64_t mSize{0};
        BufferUsage mUsage{BufferUsage::kStorage};
        bool mCpuVisible{false}; // host-visible, mapable for CPU read/write
    };

    struct ImageCreateInfo {
        ImageType mType{ImageType::k2D};
        uint32_t mWidth{1};
        uint32_t mHeight{1};
        uint32_t mDepth{1};
        uint32_t mMipLevels{1};
        uint32_t mLayerCount{1};
        Format mFormat{Format::kR8G8B8A8Unorm};
        ImageUsage mUsage{ImageUsage::kSampled};
        // Multisampling: 1 (default), 2, 4 or 8. Multisampled images can only
        // be used as attachments (resolve them into a single-sample image to
        // sample the result).
        uint32_t mSampleCount{1};
    };

    struct SamplerCreateInfo {
        Filter mMinFilter{Filter::kLinear};
        Filter mMagFilter{Filter::kLinear};
        AddressMode mAddressModeU{AddressMode::kRepeat};
        AddressMode mAddressModeV{AddressMode::kRepeat};
        AddressMode mAddressModeW{AddressMode::kRepeat};
    };

    // ---- shader reflection (backend-agnostic, extracted from .spv) ----

    struct DescriptorBindingInfo {
        uint32_t mBinding{0};
        DescriptorType mType{DescriptorType::kStorageBuffer};
        uint32_t mCount{1};
        std::string mName; // SPIR-V binding name (name-addressed binding)
    };

    struct PushConstantRange {
        ShaderStage mStage{ShaderStage::kCompute};
        uint32_t mOffset{0};
        uint32_t mSize{0};
    };

    // One member of a push constant block (reflection; name lets higher
    // layers address fields by name, e.g. an engine feeding mModel).
    struct PushConstantFieldInfo {
        std::string mName;
        uint32_t mOffset{0};
        uint32_t mSize{0};
        ShaderStage mStage{ShaderStage::kCompute};
    };

    struct VertexInputAttributeInfo {
        uint32_t mLocation{0};
        Format mFormat{Format::kR32G32B32A32Float};
    };

    struct ShaderReflection {
        std::vector<std::vector<DescriptorBindingInfo>> mDescriptorSets; // [setIndex][bindings]
        std::vector<PushConstantRange> mPushConstantRanges;
        std::vector<PushConstantFieldInfo> mPushConstantFields; // block members, per stage
        std::vector<VertexInputAttributeInfo> mVertexInputs; // vertex stage only
        uint32_t mWorkgroupSizeX{1};
        uint32_t mWorkgroupSizeY{1};
        uint32_t mWorkgroupSizeZ{1};
    };
}// namespace moe::rhi