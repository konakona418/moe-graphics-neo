#pragma once

#include <Neo/Renderer.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/Sampler.hpp>
#include <RHI/Shader.hpp>

#include <cstdint>
#include <vector>

namespace examples {
    struct BloomParams {
        float mThreshold{1.0f}; // bright-pass threshold (linear HDR)
        float mSoftKnee{0.5f};  // soft-knee width
        float mIntensity{1.0f}; // bloom contribution in the composite
    };

    // Downsample/upsample bloom (Call of Duty style): bright pass into a
    // half-resolution mip chain, then a 13-tap downsample and an additive
    // 3x3-tent upsample back up. Everything is linear HDR; the final composite
    // writes straight to the (sRGB) swapchain, so the hardware gamma-encodes.
    class Bloom {
    public:
        Bloom() = default;
        ~Bloom();
        Bloom(const Bloom&) = delete;
        Bloom& operator=(const Bloom&) = delete;

        // `width`/`height` are the scene size; the bloom's first mip is half
        // that. shaderDir must end with '/'.
        bool Init(moe::rhi::Device& device, moe::neo::Renderer& renderer, const char* shaderDir,
                uint32_t width, uint32_t height, uint32_t mipCount = 5);

        // Runs bright pass -> downsample chain -> upsample chain, then
        // composites scene + bloom to the swapchain.
        void Render(moe::neo::Renderer& renderer, const moe::rhi::Image& scene,
                const BloomParams& params);

        void Destroy();

    private:
        moe::rhi::Device* mDevice{nullptr};
        moe::neo::Renderer* mRenderer{nullptr};
        moe::rhi::Sampler mSampler;

        moe::rhi::Shader mPrefilterVert;
        moe::rhi::Shader mPrefilterFrag;
        moe::rhi::ShaderProgram mPrefilterProgram;
        moe::rhi::Shader mDownsampleVert;
        moe::rhi::Shader mDownsampleFrag;
        moe::rhi::ShaderProgram mDownsampleProgram;
        moe::rhi::Shader mUpsampleVert;
        moe::rhi::Shader mUpsampleFrag;
        moe::rhi::ShaderProgram mUpsampleProgram;
        moe::rhi::Shader mCompositeVert;
        moe::rhi::Shader mCompositeFrag;
        moe::rhi::ShaderProgram mCompositeProgram;

        std::vector<moe::neo::RenderTargetHandle> mMips;
        uint32_t mWidth{0};
        uint32_t mHeight{0};

        int32_t mPfTexelSize{-1};
        int32_t mPfThreshold{-1};
        int32_t mPfSoftKnee{-1};
        int32_t mDsTexelSize{-1};
        int32_t mUpTexelSize{-1};
        int32_t mUpRadius{-1};
        int32_t mCompIntensity{-1};
    };
}// namespace examples
