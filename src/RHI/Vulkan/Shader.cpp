#include "RHI/Shader.hpp"

#include <spirv_reflect.h>

#include "Core/FileIo.hpp"
#include "Core/Logger.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

namespace moe::rhi {
    namespace {
        uint64_t HashBytes(const char* data, size_t size) {
            uint64_t hash = 1469598103934665603ull; // FNV-1a offset basis
            for (size_t i = 0; i < size; ++i) {
                hash ^= static_cast<uint8_t>(data[i]);
                hash *= 1099511628211ull;
            }
            return hash;
        }

        DescriptorType ToDescriptorType(SpvReflectDescriptorType type) {
            switch (type) {
                case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return DescriptorType::kUniformBuffer;
                case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER: return DescriptorType::kStorageBuffer;
                case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return DescriptorType::kCombinedImageSampler;
                case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return DescriptorType::kSampledImage;
                case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE: return DescriptorType::kStorageImage;
                case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER: return DescriptorType::kSampler;
                default: {
                    MOE_RHI_ASSERT(false, "ToDescriptorType: unhandled SPIRV-Reflect descriptor type");
                    return DescriptorType::kStorageBuffer;
                }
            }
        }

        // Maps a reflected vertex-input format to the RHI Format. Formats the
        // Reflected vertex inputs are informational (the Renderer derives the
        // pipeline's vertex layout from mesh/instance data); unknown formats
        // (e.g. matrix attributes that SPIRV-Reflect splits into components)
        // degrade to undefined instead of aborting.
        Format ToFormat(SpvReflectFormat format) {
            switch (format) {
                case SPV_REFLECT_FORMAT_R16G16_SFLOAT: return Format::kR16G16Float;
                case SPV_REFLECT_FORMAT_R16G16B16A16_SFLOAT: return Format::kR16G16B16A16Float;
                case SPV_REFLECT_FORMAT_R32_SFLOAT: return Format::kR32Float;
                case SPV_REFLECT_FORMAT_R32_UINT: return Format::kR32Uint;
                case SPV_REFLECT_FORMAT_R32G32_SFLOAT: return Format::kR32G32Float;
                case SPV_REFLECT_FORMAT_R32G32B32_SFLOAT: return Format::kR32G32B32Float;
                case SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT: return Format::kR32G32B32A32Float;
                default: return Format::kUndefined;
            }
        }
    }// namespace

    Shader::Shader() : mImpl(std::make_unique<ShaderImpl>()) {}

    Shader::~Shader() = default;

    bool Shader::Load(const char* spvPath, ShaderStage stage) {
        std::vector<uint8_t> bytes;
        if (!moe::ReadFileBytes(spvPath, bytes, mImpl->mLastError)) {
            return false;
        }
        mImpl->mCode.assign(bytes.begin(), bytes.end());
        mImpl->mPath = spvPath;
        mImpl->mStage = stage;
        mImpl->mContentHash = HashBytes(mImpl->mCode.data(), mImpl->mCode.size());

        SpvReflectShaderModule module{};
        if (spvReflectCreateShaderModule(mImpl->mCode.size(), mImpl->mCode.data(), &module) != SPV_REFLECT_RESULT_SUCCESS) {
            mImpl->mLastError = std::string("SPIRV-Reflect failed to parse: ") + spvPath;
            return false;
        }

        ShaderReflection reflection;
        reflection.mDescriptorSets.resize(module.descriptor_set_count);
        for (uint32_t set = 0; set < module.descriptor_set_count; ++set) {
            const SpvReflectDescriptorSet& reflectSet = module.descriptor_sets[set];
            auto& bindings = reflection.mDescriptorSets[reflectSet.set];
            for (uint32_t b = 0; b < reflectSet.binding_count; ++b) {
                const SpvReflectDescriptorBinding* binding = reflectSet.bindings[b];
                bindings.push_back({
                        binding->binding,
                        ToDescriptorType(binding->descriptor_type),
                        binding->count,
                });
            }
        }
        for (uint32_t i = 0; i < module.push_constant_block_count; ++i) {
            const SpvReflectBlockVariable& block = module.push_constant_blocks[i];
            reflection.mPushConstantRanges.push_back({stage, block.offset, block.size});
            for (uint32_t f = 0; f < block.member_count; ++f) {
                const SpvReflectBlockVariable& member = block.members[f];
                reflection.mPushConstantFields.push_back(
                        {member.name, block.offset + member.offset, member.size, stage});
            }
        }
        for (uint32_t i = 0; i < module.input_variable_count; ++i) {
            const SpvReflectInterfaceVariable* input = module.input_variables[i];
            if (input->location == static_cast<uint32_t>(-1)) {
                continue; // builtin
            }
            reflection.mVertexInputs.push_back({input->location, ToFormat(input->format)});
        }
        for (uint32_t i = 0; i < module.entry_point_count; ++i) {
            const SpvReflectEntryPoint& entry = module.entry_points[i];
            reflection.mWorkgroupSizeX = entry.local_size.x;
            reflection.mWorkgroupSizeY = entry.local_size.y;
            reflection.mWorkgroupSizeZ = entry.local_size.z;
        }

        spvReflectDestroyShaderModule(&module);

        mImpl->mReflection = std::move(reflection);
        mImpl->mLastError.clear();
        moe::Logger::info("RHI shader loaded: {} ({} bytes)", spvPath, mImpl->mCode.size());
        return true;
    }

    bool Shader::Reload() {
        if (mImpl->mPath.empty()) {
            mImpl->mLastError = "Shader has no path to reload";
            return false;
        }
        return Load(mImpl->mPath.c_str(), mImpl->mStage);
    }

    const std::string& Shader::GetPath() const {
        return mImpl->mPath;
    }

    const std::string& Shader::GetLastError() const {
        return mImpl->mLastError;
    }

    ShaderStage Shader::GetStage() const {
        return mImpl->mStage;
    }

    const ShaderReflection& Shader::GetReflection() const {
        return mImpl->mReflection;
    }

    const std::vector<char>& Shader::GetCode() const {
        return mImpl->mCode;
    }

    uint64_t Shader::GetContentHash() const {
        return mImpl->mContentHash;
    }

    ShaderProgram::ShaderProgram() : mImpl(std::make_unique<ShaderProgramImpl>()) {}

    ShaderProgram::~ShaderProgram() = default;

    bool ShaderProgram::AddShader(const Shader& shader) {
        const uint32_t index = static_cast<uint32_t>(shader.GetStage());
        if (mImpl->mStages[index] != nullptr) {
            return false; // stage already present
        }
        mImpl->mStages[index] = &shader;
        mImpl->mStageCount++;
        return true;
    }

    const Shader* ShaderProgram::GetStage(ShaderStage stage) const {
        return mImpl->mStages[static_cast<uint32_t>(stage)];
    }

    bool ShaderProgram::HasStage(ShaderStage stage) const {
        return mImpl->mStages[static_cast<uint32_t>(stage)] != nullptr;
    }

    uint32_t ShaderProgram::GetStageCount() const {
        return mImpl->mStageCount;
    }

    uint64_t ShaderProgram::GetContentHash() const {
        uint64_t hash = 1469598103934665603ull;
        for (const Shader* stage : mImpl->mStages) {
            if (stage != nullptr) {
                hash ^= stage->GetContentHash();
                hash *= 1099511628211ull;
            }
        }
        return hash;
    }
}// namespace moe::rhi