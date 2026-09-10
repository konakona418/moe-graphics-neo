#include <RHI/Buffer.hpp>

#include <Core/Defer.hpp>
#include "TestSupport.hpp"
#include <Core/Error.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/DescriptorSet.hpp>
#include <RHI/Device.hpp>
#include <RHI/Image.hpp>
#include <RHI/Pipeline.hpp>
#include <RHI/PipelineCache.hpp>
#include <RHI/RenderGraph.hpp>
#include <RHI/Sampler.hpp>
#include <RHI/Shader.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Headless validation of the clouds pipeline: compute 3D noise -> fullscreen
// raymarch into an offscreen color target via the RenderGraph -> read back and
// assert the frame is non-black and non-uniform.
namespace {
    constexpr uint32_t kNoiseSize = 64;
    constexpr uint32_t kWidth = 1280;
    constexpr uint32_t kHeight = 720;
    constexpr uint32_t kPixelCount = kWidth * kHeight;
    const float kClear[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    struct NoisePushConstants {
        uint32_t mSize;
        float mTime;
        float mSeed;
        float mPadding;
    };

    struct CloudPushConstants {
        float mCameraPos[3];
        float mForward[3];
        float mRight[3];
        float mUp[3];
        float mSunDir[3];
        float mTanHalfFov;
        float mAspect;
        float mTime;
        float mBoxMin[3];
        float mBoxMax[3];
    };

    struct NoisePass : moe::rhi::Pass {
        moe::rhi::ComputePipeline* mPipeline{nullptr};
        moe::rhi::DescriptorSet* mSet{nullptr};
        NoisePushConstants* mPc{nullptr};

        bool Execute(moe::rhi::CommandList& cmd) override {
            cmd.SetPushConstants(*mPipeline, 0, sizeof(NoisePushConstants), mPc);
            cmd.BindDescriptorSet(*mPipeline, *mSet, 0);
            cmd.Dispatch(*mPipeline, kNoiseSize / 4, kNoiseSize / 4, kNoiseSize / 4);
            return true;
        }
    };

    struct CloudPass : moe::rhi::Pass {
        moe::rhi::GraphicsPipeline* mPipeline{nullptr};
        moe::rhi::DescriptorSet* mSet{nullptr};
        moe::rhi::Image* mColorTarget{nullptr};
        CloudPushConstants* mPc{nullptr};
        uint32_t mWidth{0};
        uint32_t mHeight{0};

        bool Execute(moe::rhi::CommandList& cmd) override {
            cmd.BeginRendering(*mColorTarget, kClear);
            cmd.BindGraphicsPipeline(*mPipeline);
            cmd.SetViewport(mWidth, mHeight);
            cmd.BindDescriptorSet(*mPipeline, *mSet, 0);
            cmd.SetPushConstants(*mPipeline, 0, sizeof(CloudPushConstants), mPc);
            cmd.Draw(3, 1, 0, 0);
            cmd.EndRendering();
            return true;
        }
    };
}// namespace

int main() {
    constexpr const char* kTestName = "Clouds smoke";
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;

    moe::rhi::Image noiseTex;
    moe::rhi::Image colorTarget;
    moe::rhi::Sampler sampler;
    moe::rhi::Shader noiseComp;
    moe::rhi::Shader vert;
    moe::rhi::Shader frag;
    moe::rhi::ShaderProgram noiseProgram;
    moe::rhi::ShaderProgram cloudProgram;
    moe::rhi::ComputePipeline noisePipeline;
    moe::rhi::GraphicsPipeline cloudPipeline;
    moe::rhi::DescriptorSetLayout noiseLayout;
    moe::rhi::DescriptorSetLayout cloudLayout;
    moe::rhi::DescriptorSet noiseSet;
    moe::rhi::DescriptorSet cloudSet;
    moe::rhi::RenderGraph graph;
    moe::rhi::Buffer readback;
    moe::rhi::CommandList commandList;

    moe::rhi::ImageCreateInfo noiseInfo{};
    moe::rhi::ImageCreateInfo targetInfo{};
    moe::rhi::GraphicsPipelineState graphicsState{};
    moe::rhi::ComputePipelineState computeState{};
    moe::rhi::SamplerCreateInfo samplerInfo{};
    moe::rhi::BufferCreateInfo readbackInfo{};
    moe::rhi::SyncInfo sync{};
    moe::rhi::ResourceId noiseId = moe::rhi::kInvalidResourceId;
    moe::rhi::ResourceId targetId = moe::rhi::kInvalidResourceId;

    NoisePushConstants noisePc{};
    CloudPushConstants cloudPc{};
    NoisePass noisePass;
    CloudPass cloudPass;


    moe::Defer cleanup([&] {
        commandList.Destroy();
        readback.Destroy();
        noiseSet.Destroy();
        cloudSet.Destroy();
        sampler.Destroy();
        noiseTex.Destroy();
        colorTarget.Destroy();
        cache.Destroy();
        device.Destroy();
    });
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }

    noiseInfo.mType = moe::rhi::ImageType::k3D;
    noiseInfo.mWidth = kNoiseSize;
    noiseInfo.mHeight = kNoiseSize;
    noiseInfo.mDepth = kNoiseSize;
    noiseInfo.mFormat = moe::rhi::Format::kR8G8B8A8Unorm;
    noiseInfo.mUsage = moe::rhi::ImageUsage::kStorage | moe::rhi::ImageUsage::kSampled
            | moe::rhi::ImageUsage::kTransferSrc;
    if (!device.CreateImage(noiseInfo, noiseTex)) {
        return moe::test::Fail(kTestName);
    }

    targetInfo.mType = moe::rhi::ImageType::k2D;
    targetInfo.mWidth = kWidth;
    targetInfo.mHeight = kHeight;
    targetInfo.mDepth = 1;
    targetInfo.mFormat = moe::rhi::Format::kR8G8B8A8Srgb;
    targetInfo.mUsage = moe::rhi::ImageUsage::kColorAttachment | moe::rhi::ImageUsage::kTransferSrc;
    if (!device.CreateImage(targetInfo, colorTarget)) {
        return moe::test::Fail(kTestName);
    }

    if (!device.CreateSampler(samplerInfo, sampler)) {
        return moe::test::Fail(kTestName);
    }

    if (!noiseComp.Load(MOE_SOURCE_DIR "/shaders/examples/clouds_noise.comp.spv", moe::rhi::ShaderStage::kCompute)
            || !vert.Load(MOE_SOURCE_DIR "/shaders/examples/clouds.vert.spv", moe::rhi::ShaderStage::kVertex)
            || !frag.Load(MOE_SOURCE_DIR "/shaders/examples/clouds.frag.spv", moe::rhi::ShaderStage::kFragment)) {
        return moe::test::Fail(kTestName, "shader load failed");
    }
    if (!noiseProgram.AddShader(noiseComp)
            || !cloudProgram.AddShader(vert) || !cloudProgram.AddShader(frag)) {
        return moe::test::Fail(kTestName, "program add failed");
    }

    computeState.mProgram = &noiseProgram;
    if (!device.GetOrCreateComputePipeline(computeState, noisePipeline)) {
        return moe::test::Fail(kTestName);
    }
    graphicsState.mProgram = &cloudProgram;
    graphicsState.mTopology = moe::rhi::PrimitiveTopology::kTriangleList;
    graphicsState.mRaster.mCullMode = moe::rhi::CullMode::kNone;
    graphicsState.mColorFormatCount = 1;
    graphicsState.mColorFormats[0] = moe::rhi::Format::kR8G8B8A8Srgb;
    graphicsState.mBlendAttachmentCount = 1;
    if (!device.GetOrCreateGraphicsPipeline(graphicsState, cloudPipeline)) {
        return moe::test::Fail(kTestName);
    }

    if (!noisePipeline.GetDescriptorSetLayout(0, noiseLayout)
            || !cloudPipeline.GetDescriptorSetLayout(0, cloudLayout)) {
        return moe::test::Fail(kTestName, "no descriptor set layout 0");
    }
    if (!device.CreateDescriptorSet(noiseLayout, noiseSet)
            || !device.CreateDescriptorSet(cloudLayout, cloudSet)) {
        return moe::test::Fail(kTestName);
    }
    if (!noiseSet.WriteImage(0, noiseTex, moe::rhi::DescriptorType::kStorageImage)
            || !cloudSet.WriteImage(0, noiseTex, moe::rhi::DescriptorType::kSampledImage)
            || !cloudSet.WriteSampler(1, sampler)) {
        return moe::test::Fail(kTestName, "descriptor write failed");
    }

    readbackInfo.mSize = sizeof(uint32_t) * kPixelCount;
    readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
    readbackInfo.mCpuVisible = true;
    if (!device.CreateBuffer(readbackInfo, readback)) {
        return moe::test::Fail(kTestName);
    }
    if (!device.CreateCommandList(commandList)) {
        return moe::test::Fail(kTestName);
    }

    noiseId = graph.RegisterImage(noiseTex);
    targetId = graph.RegisterImage(colorTarget);
    {
        moe::rhi::PassDesc desc{};
        desc.mName = "noise";
        desc.mPass = &noisePass;
        moe::rhi::ResourceAccess access{};
        access.mResource = noiseId;
        access.mIsWrite = true;
        access.mStage = moe::rhi::PipelineStage::kComputeShader;
        access.mAccess = moe::rhi::Access::kShaderWrite;
        access.mLayout = moe::rhi::ImageLayout::kGeneral;
        desc.mWrites.push_back(access);
        if (!graph.AddPass(desc)) {
            return moe::test::Fail(kTestName, "add noise pass failed");
        }
    }
    {
        moe::rhi::PassDesc desc{};
        desc.mName = "raymarch";
        desc.mPass = &cloudPass;
        moe::rhi::ResourceAccess access{};
        access.mResource = noiseId;
        access.mIsWrite = false;
        access.mStage = moe::rhi::PipelineStage::kFragmentShader;
        access.mAccess = moe::rhi::Access::kShaderRead;
        access.mLayout = moe::rhi::ImageLayout::kShaderReadOnly;
        desc.mReads.push_back(access);
        access.mResource = targetId;
        access.mIsWrite = true;
        access.mStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
        access.mAccess = moe::rhi::Access::kColorAttachmentWrite;
        access.mLayout = moe::rhi::ImageLayout::kColorAttachment;
        desc.mWrites.push_back(access);
        if (!graph.AddPass(desc)) {
            return moe::test::Fail(kTestName, "add raymarch pass failed");
        }
    }
    if (!graph.Compile()) {
        return moe::test::Fail(kTestName, "graph compile failed");
    }

    noisePc.mSize = kNoiseSize;
    noisePc.mTime = 1.0f;
    noisePc.mSeed = 13.7f;
    // demo's orbit camera at frame ~0
    cloudPc.mCameraPos[0] = 14.0f;
    cloudPc.mCameraPos[1] = 4.0f;
    cloudPc.mCameraPos[2] = 0.056f;
    cloudPc.mForward[0] = -0.9843f;
    cloudPc.mForward[1] = -0.1758f;
    cloudPc.mForward[2] = -0.0039f;
    cloudPc.mRight[0] = -0.0040f;
    cloudPc.mRight[1] = 0.0f;
    cloudPc.mRight[2] = 0.99999f;
    cloudPc.mUp[0] = -0.1758f;
    cloudPc.mUp[1] = 0.9843f;
    cloudPc.mUp[2] = -0.0007f;
    cloudPc.mSunDir[0] = 0.4f;
    cloudPc.mSunDir[1] = 0.55f;
    cloudPc.mSunDir[2] = -0.6f;
    cloudPc.mTanHalfFov = 0.57735f;
    cloudPc.mAspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);
    cloudPc.mTime = 1.0f;
    cloudPc.mBoxMin[0] = -10.0f;
    cloudPc.mBoxMin[1] = -1.0f;
    cloudPc.mBoxMin[2] = -10.0f;
    cloudPc.mBoxMax[0] = 10.0f;
    cloudPc.mBoxMax[1] = 3.5f;
    cloudPc.mBoxMax[2] = 10.0f;
    noisePass.mPipeline = &noisePipeline;
    noisePass.mSet = &noiseSet;
    noisePass.mPc = &noisePc;
    cloudPass.mPipeline = &cloudPipeline;
    cloudPass.mSet = &cloudSet;
    cloudPass.mColorTarget = &colorTarget;
    cloudPass.mPc = &cloudPc;
    cloudPass.mWidth = kWidth;
    cloudPass.mHeight = kHeight;

    commandList.Begin();
    if (!graph.Execute(commandList)) {
        return moe::test::Fail(kTestName, "graph execute failed");
    }
    sync.mSrcStage = moe::rhi::PipelineStage::kColorAttachmentOutput;
    sync.mSrcAccess = moe::rhi::Access::kColorAttachmentWrite;
    sync.mDstStage = moe::rhi::PipelineStage::kTransfer;
    sync.mDstAccess = moe::rhi::Access::kTransferRead;
    commandList.ImageBarrier(colorTarget, moe::rhi::ImageLayout::kColorAttachment,
            moe::rhi::ImageLayout::kTransferSrc, sync);
    commandList.CopyImageToBuffer(colorTarget, readback);
    commandList.End();
    if (!device.Submit(commandList, true)) {
        return moe::test::Fail(kTestName);
    }

    {
        auto* data = static_cast<uint32_t*>(readback.Map());
        if (!data) {
            return moe::test::Fail(kTestName, "failed to map readback");
        }
        uint32_t minV = 0xFFFFFFFFu;
        uint32_t maxV = 0u;
        uint64_t sum = 0;
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            minV = data[i] < minV ? data[i] : minV;
            maxV = data[i] > maxV ? data[i] : maxV;
            sum += data[i];
        }
        readback.Unmap();
        const uint64_t avg = sum / kPixelCount;
        std::printf("Clouds smoke: min=0x%08X max=0x%08X avg=0x%08lX\n", minV, maxV, (unsigned long) avg);
        if (maxV == 0u || minV == maxV) {
            return moe::test::Fail(kTestName, "raymarch produced a black or uniform frame");
        }
    }

    std::printf("Clouds smoke passed.\n");

    return EXIT_SUCCESS;
}
