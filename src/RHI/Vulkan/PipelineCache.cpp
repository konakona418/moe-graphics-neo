#include "RHI/PipelineCache.hpp"

#include "RHI/CommandList.hpp"
#include "RHI/DescriptorSet.hpp"
#include "RHI/Device.hpp"
#include "RHI/Shader.hpp"
#include "Core/Logger.hpp"
#include "Mappings.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace moe::rhi {
    namespace {
        VkShaderModule CreateShaderModule(VkDevice device, const Shader& shader) {
            const std::vector<char>& code = shader.GetCode();
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            return module;
        }

        bool CreateDescriptorSetLayout(VkDevice device,
                const std::vector<DescriptorBindingInfo>& bindings, VkDescriptorSetLayout& outLayout) {
            std::vector<VkDescriptorSetLayoutBinding> vkBindings;
            std::vector<VkDescriptorBindingFlags> bindingFlags;
            vkBindings.reserve(bindings.size());
            bindingFlags.reserve(bindings.size());
            for (const auto& binding : bindings) {
                vkBindings.push_back({
                        binding.mBinding,
                        ToVkDescriptorType(binding.mType),
                        binding.mCount,
                        VK_SHADER_STAGE_ALL,
                        nullptr,
                });
                // Unbounded descriptor arrays (bindless, e.g. Texture2D tex[])
                // reflect with count 0: they must be declared variable +
                // partially bound, otherwise the layout cannot be created.
                VkDescriptorBindingFlags flags = 0;
                if (binding.mCount == 0) {
                    flags = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
                            | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
                }
                bindingFlags.push_back(flags);
            }
            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(vkBindings.size());
            layoutInfo.pBindings = vkBindings.data();
            VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
            if (std::any_of(bindingFlags.begin(), bindingFlags.end(),
                        [](VkDescriptorBindingFlags f) { return f != 0; })) {
                flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                flagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
                flagsInfo.pBindingFlags = bindingFlags.data();
                layoutInfo.pNext = &flagsInfo;
            }
            return vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &outLayout) == VK_SUCCESS;
        }

        // Merges per-set bindings from all stages into a single per-set map.
        void MergeDescriptorSets(const ShaderProgram& program,
                std::vector<std::vector<DescriptorBindingInfo>>& outSets) {
            outSets.clear();
            for (uint32_t stageIndex = 0; stageIndex < 4; ++stageIndex) {
                const Shader* shader = program.GetStage(static_cast<ShaderStage>(stageIndex));
                if (shader == nullptr) {
                    continue;
                }
                const auto& sets = shader->GetReflection().mDescriptorSets;
                if (sets.size() > outSets.size()) {
                    outSets.resize(sets.size());
                }
                for (size_t setIndex = 0; setIndex < sets.size(); ++setIndex) {
                    for (const auto& binding : sets[setIndex]) {
                        bool exists = false;
                        for (auto& existing : outSets[setIndex]) {
                            if (existing.mBinding == binding.mBinding) {
                                exists = true;
                                break;
                            }
                        }
                        if (!exists) {
                            outSets[setIndex].push_back(binding);
                        }
                    }
                }
            }
        }

        void MergePushConstantRanges(const ShaderProgram& program,
                std::vector<VkPushConstantRange>& outRanges) {
            outRanges.clear();
            for (uint32_t stageIndex = 0; stageIndex < 4; ++stageIndex) {
                const Shader* shader = program.GetStage(static_cast<ShaderStage>(stageIndex));
                if (shader == nullptr) {
                    continue;
                }
                for (const auto& range : shader->GetReflection().mPushConstantRanges) {
                    bool merged = false;
                    for (auto& existing : outRanges) {
                        if (existing.offset == range.mOffset && existing.size == range.mSize) {
                            existing.stageFlags |= ToVkShaderStage(range.mStage);
                            merged = true;
                            break;
                        }
                    }
                    if (!merged) {
                        outRanges.push_back({ToVkShaderStage(range.mStage), range.mOffset, range.mSize});
                    }
                }
            }
        }

        bool BuildPipelineLayout(DeviceImpl* device, const ShaderProgram& program,
                VkPipelineLayout& outLayout, std::vector<VkDescriptorSetLayout>& outSetLayouts,
                std::vector<std::vector<DescriptorBindingInfo>>& outSetBindings,
                VkShaderStageFlags& outPushStages) {
            std::vector<std::vector<DescriptorBindingInfo>> setBindings;
            MergeDescriptorSets(program, setBindings);

            outSetLayouts.clear();
            for (const auto& bindings : setBindings) {
                VkDescriptorSetLayout layout;
                if (!CreateDescriptorSetLayout(device->mDevice, bindings, layout)) {
                    return false;
                }
                outSetLayouts.push_back(layout);
            }
            outSetBindings = std::move(setBindings);

            std::vector<VkPushConstantRange> pushRanges;
            MergePushConstantRanges(program, pushRanges);

            outPushStages = 0;
            for (const auto& range : pushRanges) {
                outPushStages |= range.stageFlags;
            }

            VkPipelineLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layoutInfo.setLayoutCount = static_cast<uint32_t>(outSetLayouts.size());
            layoutInfo.pSetLayouts = outSetLayouts.data();
            layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushRanges.size());
            layoutInfo.pPushConstantRanges = pushRanges.data();
            return vkCreatePipelineLayout(device->mDevice, &layoutInfo, nullptr, &outLayout) == VK_SUCCESS;
        }

        bool BuildComputeNode(DeviceImpl* device, VkPipelineCache driverCache,
                const ComputePipelineState& state, PipelineNode& node) {
            const Shader* compute = state.mProgram->GetStage(ShaderStage::kCompute);
            if (compute == nullptr) {
                return false;
            }
            if (!BuildPipelineLayout(device, *state.mProgram, node.mPipelineLayout,
                        node.mSetLayouts, node.mSetBindings, node.mPushConstantStages)) {
                return false;
            }
            VkShaderModule module = CreateShaderModule(device->mDevice, *compute);
            if (module == VK_NULL_HANDLE) {
                return false;
            }

            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = module;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = node.mPipelineLayout;
            const VkResult result = vkCreateComputePipelines(device->mDevice,
                    driverCache, 1, &pipelineInfo, nullptr, &node.mPipeline);
            vkDestroyShaderModule(device->mDevice, module, nullptr);
            return result == VK_SUCCESS;
        }

        bool BuildGraphicsNode(DeviceImpl* device, VkPipelineCache driverCache,
                const GraphicsPipelineState& state, PipelineNode& node) {
            if (!BuildPipelineLayout(device, *state.mProgram, node.mPipelineLayout,
                        node.mSetLayouts, node.mSetBindings, node.mPushConstantStages)) {
                return false;
            }

            std::vector<VkShaderModule> modules;
            std::vector<VkPipelineShaderStageCreateInfo> stages;
            for (uint32_t stageIndex = 0; stageIndex < 4; ++stageIndex) {
                const Shader* shader = state.mProgram->GetStage(static_cast<ShaderStage>(stageIndex));
                if (shader == nullptr) {
                    continue;
                }
                VkShaderModule module = CreateShaderModule(device->mDevice, *shader);
                if (module == VK_NULL_HANDLE) {
                    continue;
                }
                modules.push_back(module);
                stages.push_back({});
                auto& stageInfo = stages.back();
                stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stageInfo.stage = ToVkShaderStage(shader->GetStage());
                stageInfo.module = module;
                stageInfo.pName = "main";
            }

            VkPipelineVertexInputStateCreateInfo vertexInput{};
            vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            std::vector<VkVertexInputBindingDescription> bindings;
            std::vector<VkVertexInputAttributeDescription> attributes;
            for (uint32_t i = 0; i < state.mVertexBindingCount; ++i) {
                const auto& b = state.mVertexBindings[i];
                bindings.push_back({b.mBinding, b.mStride,
                        b.mPerInstance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
            }
            for (uint32_t i = 0; i < state.mVertexAttributeCount; ++i) {
                const auto& a = state.mVertexAttributes[i];
                attributes.push_back({a.mLocation, a.mBinding, ToVkFormat(a.mFormat), a.mOffset});
            }
            vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
            vertexInput.pVertexBindingDescriptions = bindings.data();
            vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
            vertexInput.pVertexAttributeDescriptions = attributes.data();

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            switch (state.mTopology) {
                case PrimitiveTopology::kPointList: inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
                case PrimitiveTopology::kLineList: inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
                case PrimitiveTopology::kTriangleList: inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
                case PrimitiveTopology::kTriangleStrip: inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
                default:
                    MOE_RHI_ASSERT(false, "BuildGraphicsNode: unhandled PrimitiveTopology");
            }

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.depthClampEnable = state.mRaster.mDepthClampEnable;
            switch (state.mRaster.mPolygonMode) {
                case PolygonMode::kFill: raster.polygonMode = VK_POLYGON_MODE_FILL; break;
                case PolygonMode::kLine: raster.polygonMode = VK_POLYGON_MODE_LINE; break;
                case PolygonMode::kPoint: raster.polygonMode = VK_POLYGON_MODE_POINT; break;
                default:
                    MOE_RHI_ASSERT(false, "BuildGraphicsNode: unhandled PolygonMode");
            }
            switch (state.mRaster.mCullMode) {
                case CullMode::kNone: raster.cullMode = VK_CULL_MODE_NONE; break;
                case CullMode::kFront: raster.cullMode = VK_CULL_MODE_FRONT_BIT; break;
                case CullMode::kBack: raster.cullMode = VK_CULL_MODE_BACK_BIT; break;
                default:
                    MOE_RHI_ASSERT(false, "BuildGraphicsNode: unhandled CullMode");
            }
            switch (state.mRaster.mFrontFace) {
                case FrontFace::kCounterClockwise: raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; break;
                case FrontFace::kClockwise: raster.frontFace = VK_FRONT_FACE_CLOCKWISE; break;
                default:
                    MOE_RHI_ASSERT(false, "BuildGraphicsNode: unhandled FrontFace");
            }
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = static_cast<VkSampleCountFlagBits>(state.mMultisample.mSampleCount);
            multisample.sampleShadingEnable = state.mMultisample.mSampleShading;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = state.mDepth.mTestEnable;
            depthStencil.depthWriteEnable = state.mDepth.mWriteEnable;
            const VkCompareOp depthCompareOps[] = {
                    VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS, VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL,
                    VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS,
            };
            depthStencil.depthCompareOp = depthCompareOps[static_cast<uint32_t>(state.mDepth.mCompareOp)];
            const VkStencilOp stencilOps[] = {
                    VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    VK_STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_INVERT, VK_STENCIL_OP_INCREMENT_AND_WRAP,
                    VK_STENCIL_OP_DECREMENT_AND_WRAP,
            };
            const VkCompareOp stencilCompareOps[] = {
                    VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS, VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL,
                    VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS,
            };
            depthStencil.front.failOp = stencilOps[static_cast<uint32_t>(state.mStencil.mFailOp)];
            depthStencil.front.passOp = stencilOps[static_cast<uint32_t>(state.mStencil.mPassOp)];
            depthStencil.front.depthFailOp = stencilOps[static_cast<uint32_t>(state.mStencil.mDepthFailOp)];
            depthStencil.front.compareOp = stencilCompareOps[static_cast<uint32_t>(state.mStencil.mCompareOp)];
            depthStencil.front.compareMask = state.mStencil.mCompareMask;
            depthStencil.front.writeMask = state.mStencil.mWriteMask;
            depthStencil.back = depthStencil.front;

            std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
            for (uint32_t i = 0; i < state.mBlendAttachmentCount; ++i) {
                const auto& b = state.mBlendAttachments[i];
                const VkBlendFactor blendFactors[] = {
                        VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR,
                        VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_DST_COLOR,
                        VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_SRC_ALPHA,
                        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA,
                        VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                };
                const VkBlendOp blendOps[] = {
                        VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT,
                        VK_BLEND_OP_MIN, VK_BLEND_OP_MAX,
                };
                blendAttachments.push_back({
                        b.mBlendEnabled,
                        blendFactors[static_cast<uint32_t>(b.mSrcColor)],
                        blendFactors[static_cast<uint32_t>(b.mDstColor)],
                        blendOps[static_cast<uint32_t>(b.mColorOp)],
                        blendFactors[static_cast<uint32_t>(b.mSrcAlpha)],
                        blendFactors[static_cast<uint32_t>(b.mDstAlpha)],
                        blendOps[static_cast<uint32_t>(b.mAlphaOp)],
                        0xf, // colorWriteMask
                });
            }

            VkPipelineColorBlendStateCreateInfo colorBlend{};
            colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlend.attachmentCount = static_cast<uint32_t>(blendAttachments.size());
            colorBlend.pAttachments = blendAttachments.data();

            std::vector<VkFormat> colorFormats;
            for (uint32_t i = 0; i < state.mColorFormatCount; ++i) {
                colorFormats.push_back(ToVkFormat(state.mColorFormats[i]));
            }

            VkPipelineRenderingCreateInfo renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
            renderingInfo.pColorAttachmentFormats = colorFormats.data();
            renderingInfo.depthAttachmentFormat = ToVkFormat(state.mDepthFormat);

            // viewport/scissor are dynamic (set per draw via CommandList::SetViewport)
            const VkViewport viewport{0, 0, 1, 1, 0, 1};
            const VkRect2D scissor{{0, 0}, {1, 1}};
            VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

            const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamicState{};
            dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.pNext = &renderingInfo;
            pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
            pipelineInfo.pStages = stages.data();
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &raster;
            pipelineInfo.pMultisampleState = &multisample;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlend;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = node.mPipelineLayout;

            const VkResult result = vkCreateGraphicsPipelines(device->mDevice,
                    driverCache, 1, &pipelineInfo, nullptr, &node.mPipeline);
            for (auto module : modules) {
                vkDestroyShaderModule(device->mDevice, module, nullptr);
            }
            return result == VK_SUCCESS;
        }
    }// namespace

    struct DefaultPipelineCache::Impl {
        DeviceImpl* mDevice{nullptr};
        VkPipelineCache mDriverCache{VK_NULL_HANDLE};
        std::vector<std::unique_ptr<PipelineNode>> mNodes;
        std::unordered_map<uint64_t, std::vector<uint32_t>> mIndexByHash;
    };

    DefaultPipelineCache::DefaultPipelineCache() = default;

    DefaultPipelineCache::~DefaultPipelineCache() {
        MOE_RHI_ASSERT(mImpl == nullptr, "DefaultPipelineCache leaked: Destroy() not called");
    }

    bool DefaultPipelineCache::Create(Device& device) {
        if (mImpl != nullptr) {
            return false; // already bound to a device
        }
        if (!device.mImpl || device.mImpl->mDevice == VK_NULL_HANDLE) {
            return false;
        }
        mImpl = std::make_unique<Impl>();
        mImpl->mDevice = device.mImpl.get();

        VkPipelineCacheCreateInfo cacheInfo{};
        cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(mImpl->mDevice->mDevice, &cacheInfo, nullptr, &mImpl->mDriverCache) != VK_SUCCESS) {
            mImpl.reset();
            return false;
        }
        return true;
    }

    void DefaultPipelineCache::Destroy() {
        if (mImpl == nullptr) {
            return;
        }
        if (mImpl->mDevice != nullptr) {
            Clear();
            if (mImpl->mDriverCache != VK_NULL_HANDLE) {
                vkDestroyPipelineCache(mImpl->mDevice->mDevice, mImpl->mDriverCache, nullptr);
            }
        }
        mImpl.reset();
    }

    const char* DefaultPipelineCache::GetName() const {
        return "DefaultPipelineCache";
    }

    uint32_t DefaultPipelineCache::GetNodeCount() const {
        return mImpl ? static_cast<uint32_t>(mImpl->mNodes.size()) : 0;
    }

    bool DefaultPipelineCache::GetOrCreateGraphics(const GraphicsPipelineState& state, GraphicsPipeline& out) {
        if (mImpl == nullptr) {
            return false; // Create() not called
        }
        const uint64_t key = state.GetHash();
        auto it = mImpl->mIndexByHash.find(key);
        if (it != mImpl->mIndexByHash.end()) {
            for (uint32_t index : it->second) {
                PipelineNode& node = *mImpl->mNodes[index];
                if (!node.mIsCompute && node.mGraphicsState == state) {
                    out.mNode = &node;
                    return true;
                }
            }
        }

        auto node = std::make_unique<PipelineNode>();
        node->mIsCompute = false;
        node->mGraphicsState = state;
        node->mKeyHash = key;
        if (!BuildGraphicsNode(mImpl->mDevice, mImpl->mDriverCache, state, *node)) {
            return false;
        }

        const uint32_t index = static_cast<uint32_t>(mImpl->mNodes.size());
        out.mNode = node.get();
        mImpl->mNodes.push_back(std::move(node));
        mImpl->mIndexByHash[key].push_back(index);
        moe::Logger::info("RHI compiled graphics pipeline ({} stage(s), hash {:016x})",
                state.mProgram->GetStageCount(), key);
        return true;
    }

    bool DefaultPipelineCache::GetOrCreateCompute(const ComputePipelineState& state, ComputePipeline& out) {
        if (mImpl == nullptr) {
            return false; // Create() not called
        }
        const uint64_t key = state.GetHash();
        auto it = mImpl->mIndexByHash.find(key);
        if (it != mImpl->mIndexByHash.end()) {
            for (uint32_t index : it->second) {
                PipelineNode& node = *mImpl->mNodes[index];
                if (node.mIsCompute && node.mComputeState == state) {
                    out.mNode = &node;
                    return true;
                }
            }
        }

        auto node = std::make_unique<PipelineNode>();
        node->mIsCompute = true;
        node->mComputeState = state;
        node->mKeyHash = key;
        if (!BuildComputeNode(mImpl->mDevice, mImpl->mDriverCache, state, *node)) {
            return false;
        }

        const uint32_t index = static_cast<uint32_t>(mImpl->mNodes.size());
        out.mNode = node.get();
        mImpl->mNodes.push_back(std::move(node));
        mImpl->mIndexByHash[key].push_back(index);
        moe::Logger::info("RHI compiled compute pipeline ({} stage(s), hash {:016x})",
                state.mProgram->GetStageCount(), key);
        return true;
    }

    bool DefaultPipelineCache::Reload(ShaderProgram& program) {
        if (mImpl == nullptr || mImpl->mDevice == nullptr) {
            return false;
        }

        bool ok = true;
        for (uint32_t stageIndex = 0; stageIndex < 4; ++stageIndex) {
            Shader* shader = const_cast<Shader*>(program.GetStage(static_cast<ShaderStage>(stageIndex)));
            if (shader != nullptr && !shader->Reload()) {
                ok = false;
            }
        }

        // Invalidate every cached pipeline built from this program.
        for (size_t i = 0; i < mImpl->mNodes.size(); ++i) {
            PipelineNode& node = *mImpl->mNodes[i];
            const bool matches = node.mIsCompute
                    ? node.mComputeState.mProgram == &program
                    : node.mGraphicsState.mProgram == &program;
            if (!matches) {
                continue;
            }
            DeferredDeletion deletion;
            deletion.mPipeline = node.mPipeline;
            deletion.mPipelineLayout = node.mPipelineLayout;
            deletion.mSetLayouts = std::move(node.mSetLayouts);
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
            node.mPipeline = VK_NULL_HANDLE;
            node.mPipelineLayout = VK_NULL_HANDLE;

            auto& indexList = mImpl->mIndexByHash[node.mKeyHash];
            indexList.erase(std::remove(indexList.begin(), indexList.end(), static_cast<uint32_t>(i)), indexList.end());
            node.mKeyHash = 0; // mark dead; the node stays in mNodes as a tombstone
        }
        moe::Logger::info("RHI reloaded shader program, invalidated cached pipelines");
        return ok;
    }

    void DefaultPipelineCache::Clear() {
        if (mImpl->mDevice == nullptr) {
            mImpl->mNodes.clear();
            mImpl->mIndexByHash.clear();
            return;
        }
        for (auto& node : mImpl->mNodes) {
            if (node->mPipeline == VK_NULL_HANDLE) {
                continue;
            }
            DeferredDeletion deletion;
            deletion.mPipeline = node->mPipeline;
            deletion.mPipelineLayout = node->mPipelineLayout;
            deletion.mSetLayouts = std::move(node->mSetLayouts);
            mImpl->mDevice->EnqueueDeferred(std::move(deletion));
            node->mPipeline = VK_NULL_HANDLE;
            node->mPipelineLayout = VK_NULL_HANDLE;
        }
        mImpl->mNodes.clear();
        mImpl->mIndexByHash.clear();
    }
}// namespace moe::rhi