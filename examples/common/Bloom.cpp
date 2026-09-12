#include "examples/common/Bloom.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <string>

namespace examples {
    namespace {
        moe::neo::DrawState OpaqueState() {
            moe::neo::DrawState state;
            state.mDepthTest = false;
            state.mDepthWrite = false;
            state.mCullMode = moe::rhi::CullMode::kNone;
            return state;
        }

        moe::neo::DrawState AdditiveState() {
            moe::neo::DrawState state = OpaqueState();
            state.mBlendEnabled = true;
            state.mBlendSrcColor = moe::rhi::BlendFactor::kOne;
            state.mBlendDstColor = moe::rhi::BlendFactor::kOne;
            state.mBlendSrcAlpha = moe::rhi::BlendFactor::kOne;
            state.mBlendDstAlpha = moe::rhi::BlendFactor::kOne;
            return state;
        }
    }// namespace

    Bloom::~Bloom() = default;

    bool Bloom::Init(moe::rhi::Device& device, moe::neo::Renderer& renderer, const char* shaderDir,
            uint32_t width, uint32_t height, uint32_t mipCount) {
        mDevice = &device;
        mRenderer = &renderer;
        mWidth = width;
        mHeight = height;

        moe::rhi::SamplerCreateInfo samplerInfo{};
        samplerInfo.mMinFilter = moe::rhi::Filter::kLinear;
        samplerInfo.mMagFilter = moe::rhi::Filter::kLinear;
        samplerInfo.mAddressModeU = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeV = moe::rhi::AddressMode::kClampToEdge;
        samplerInfo.mAddressModeW = moe::rhi::AddressMode::kClampToEdge;
        if (!device.CreateSampler(samplerInfo, mSampler)) {
            return false;
        }

        const std::string base(shaderDir);
        auto load = [&](const char* vert, const char* frag, moe::rhi::Shader& vertShader,
                            moe::rhi::Shader& fragShader, moe::rhi::ShaderProgram& program) {
            return vertShader.Load((base + vert).c_str(), moe::rhi::ShaderStage::kVertex)
                    && fragShader.Load((base + frag).c_str(), moe::rhi::ShaderStage::kFragment)
                    && program.AddShader(vertShader) && program.AddShader(fragShader);
        };

        if (!load("bloom_prefilter.vert.spv", "bloom_prefilter.frag.spv", mPrefilterVert,
                    mPrefilterFrag, mPrefilterProgram)
                || !load("bloom_downsample.vert.spv", "bloom_downsample.frag.spv", mDownsampleVert,
                        mDownsampleFrag, mDownsampleProgram)
                || !load("bloom_upsample.vert.spv", "bloom_upsample.frag.spv", mUpsampleVert,
                        mUpsampleFrag, mUpsampleProgram)
                || !load("bloom_composite.vert.spv", "bloom_composite.frag.spv", mCompositeVert,
                        mCompositeFrag, mCompositeProgram)) {
            return false;
        }

        mPfTexelSize = renderer.GetPushConstant(mPrefilterProgram, "mTexelSize");
        mPfThreshold = renderer.GetPushConstant(mPrefilterProgram, "mThreshold");
        mPfSoftKnee = renderer.GetPushConstant(mPrefilterProgram, "mSoftKnee");
        mDsTexelSize = renderer.GetPushConstant(mDownsampleProgram, "mTexelSize");
        mUpTexelSize = renderer.GetPushConstant(mUpsampleProgram, "mTexelSize");
        mUpRadius = renderer.GetPushConstant(mUpsampleProgram, "mRadius");
        mCompIntensity = renderer.GetPushConstant(mCompositeProgram, "mIntensity");
        if (mPfTexelSize < 0 || mPfThreshold < 0 || mPfSoftKnee < 0 || mDsTexelSize < 0
                || mUpTexelSize < 0 || mUpRadius < 0 || mCompIntensity < 0) {
            return false;
        }

        mMips.reserve(mipCount);
        for (uint32_t i = 0; i < mipCount; ++i) {
            const uint32_t w = std::max(1u, width >> (i + 1));
            const uint32_t h = std::max(1u, height >> (i + 1));
            const moe::neo::RenderTargetHandle handle = renderer.CreateRenderTarget(
                    w, h, moe::rhi::Format::kR16G16B16A16Float, false, 1);
            if (!handle.IsValid()) {
                return false;
            }
            mMips.push_back(handle);
        }
        return true;
    }

    void Bloom::Render(moe::neo::Renderer& renderer, const moe::rhi::Image& scene,
            const BloomParams& params) {
        if (mMips.empty()) {
            return;
        }

        const uint32_t mipCount = static_cast<uint32_t>(mMips.size());
        auto mipSize = [&](uint32_t i) {
            return glm::vec2(static_cast<float>(std::max(1u, mWidth >> (i + 1))),
                    static_cast<float>(std::max(1u, mHeight >> (i + 1))));
        };
        auto texelOf = [](const glm::vec2& size) {
            return glm::vec2(1.0f / size.x, 1.0f / size.y);
        };

        // 1. Bright pass: scene -> mip[0].
        {
            const glm::vec2 texel = texelOf(glm::vec2(mWidth, mHeight));
            const moe::neo::PassDesc desc{"bloom-prefilter",
                    moe::neo::ColorAttachment(mMips[0], moe::rhi::LoadOp::kClear), {}};
            renderer.Execute(desc, [&](moe::neo::PassContext& pass) {
                pass.SetState(OpaqueState());
                pass.SetPushConstant(mPfTexelSize, &texel, sizeof(texel));
                pass.SetPushConstant(mPfThreshold, &params.mThreshold, sizeof(params.mThreshold));
                pass.SetPushConstant(mPfSoftKnee, &params.mSoftKnee, sizeof(params.mSoftKnee));
                pass.BindImage(0, scene);
                pass.BindSampler(1, mSampler);
                pass.DrawFullscreen(mPrefilterProgram);
            });
        }

        // 2. Downsample chain: mip[i-1] -> mip[i].
        for (uint32_t i = 1; i < mipCount; ++i) {
            const glm::vec2 texel = texelOf(mipSize(i - 1));
            moe::neo::RenderTarget* src = renderer.GetRenderTarget(mMips[i - 1]);
            const moe::neo::PassDesc desc{"bloom-downsample",
                    moe::neo::ColorAttachment(mMips[i], moe::rhi::LoadOp::kClear), {}};
            renderer.Execute(desc, [&](moe::neo::PassContext& pass) {
                pass.SetState(OpaqueState());
                pass.SetPushConstant(mDsTexelSize, &texel, sizeof(texel));
                pass.BindImage(0, *src->mImage);
                pass.BindSampler(1, mSampler);
                pass.DrawFullscreen(mDownsampleProgram);
            });
        }

        // 3. Upsample chain: mip[i] -> mip[i-1], additively blended.
        for (uint32_t i = mipCount - 1; i >= 1; --i) {
            const glm::vec2 texel = texelOf(mipSize(i));
            const float radius = 1.0f;
            moe::neo::RenderTarget* src = renderer.GetRenderTarget(mMips[i]);
            const moe::neo::PassDesc desc{"bloom-upsample",
                    moe::neo::ColorAttachment(mMips[i - 1], moe::rhi::LoadOp::kLoad), {}};
            renderer.Execute(desc, [&](moe::neo::PassContext& pass) {
                pass.SetState(AdditiveState());
                pass.SetPushConstant(mUpTexelSize, &texel, sizeof(texel));
                pass.SetPushConstant(mUpRadius, &radius, sizeof(radius));
                pass.BindImage(0, *src->mImage);
                pass.BindSampler(1, mSampler);
                pass.DrawFullscreen(mUpsampleProgram);
            });
        }

        // 4. Composite scene + mip[0] to the swapchain.
        {
            moe::neo::RenderTarget* bloom = renderer.GetRenderTarget(mMips[0]);
            const moe::neo::PassDesc desc{"bloom-composite",
                    moe::neo::ColorAttachment(moe::neo::RenderTargetHandle{},
                            moe::rhi::LoadOp::kClear),
                    {}};
            renderer.Execute(desc, [&](moe::neo::PassContext& pass) {
                pass.SetState(OpaqueState());
                pass.SetPushConstant(mCompIntensity, &params.mIntensity, sizeof(params.mIntensity));
                pass.BindImage(0, scene);
                pass.BindSampler(1, mSampler);
                pass.BindImage(2, *bloom->mImage);
                pass.DrawFullscreen(mCompositeProgram);
            });
        }
    }

    void Bloom::Destroy() {
        if (mRenderer != nullptr) {
            for (const moe::neo::RenderTargetHandle handle : mMips) {
                mRenderer->DestroyRenderTarget(handle);
            }
        }
        mMips.clear();
        mSampler.Destroy();
        mDevice = nullptr;
        mRenderer = nullptr;
    }
}// namespace examples
