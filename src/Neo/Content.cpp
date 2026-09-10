#include "Neo/Renderer.hpp"

#include <Core/Logger.hpp>

#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace moe::neo {
    namespace {
        // Warnings here would repeat every frame (names resolve per draw), so
        // each distinct message is logged once.
        void WarnOnce(const std::string& message) {
            static std::vector<std::string> warned;
            for (const std::string& entry : warned) {
                if (entry == message) {
                    return;
                }
            }
            warned.push_back(message);
            moe::Logger::warn("{}", message);
        }

        // Same, at debug level: a material override absent from the current
        // program is normal when one material is drawn by several programs
        // (e.g. the outline pass), so it must not alarm users.
        void DebugOnce(const std::string& message) {
            static std::vector<std::string> logged;
            for (const std::string& entry : logged) {
                if (entry == message) {
                    return;
                }
            }
            logged.push_back(message);
            moe::Logger::debug("{}", message);
        }

        // Descriptor binding lookup by SPIR-V name (set 0, any stage).
        const rhi::DescriptorBindingInfo* FindBinding(const rhi::ShaderProgram& program,
                const char* name) {
            const rhi::ShaderStage stages[2] = {rhi::ShaderStage::kVertex, rhi::ShaderStage::kFragment};
            for (const rhi::ShaderStage stage : stages) {
                const rhi::Shader* shader = program.GetStage(stage);
                if (shader == nullptr) {
                    continue;
                }
                const auto& sets = shader->GetReflection().mDescriptorSets;
                if (sets.empty()) {
                    continue;
                }
                for (const auto& binding : sets[0]) {
                    if (binding.mName == name) {
                        return &binding;
                    }
                }
            }
            return nullptr;
        }

        void BindNamedTexture(PassContext& pass, const rhi::ShaderProgram& program, const char* name,
                const UploadedTexture& texture) {
            const rhi::DescriptorBindingInfo* image = FindBinding(program, name);
            if (image == nullptr) {
                return;
            }
            pass.BindImage(image->mBinding, texture.mImage);
            // sampler convention: the sampler for texture `X` is named `XSampler`
            const std::string samplerName = std::string(name) + "Sampler";
            const rhi::DescriptorBindingInfo* sampler = FindBinding(program, samplerName.c_str());
            if (sampler != nullptr) {
                pass.BindSampler(sampler->mBinding, texture.mSampler);
            }
        }

        void FeedMat4(PassContext& pass, const rhi::ShaderProgram& program, const char* name,
                const glm::mat4& value) {
            const int32_t index = pass.GetPushConstant(program, name);
            if (index < 0) {
                return;
            }
            if (pass.GetPushConstantSize(index) != sizeof(glm::mat4)) {
                WarnOnce(std::string("DrawModel: push constant '") + name + "' is not a mat4; skipped");
                return;
            }
            pass.SetPushConstant(index, &value, sizeof(value));
        }

        void FeedVec3(PassContext& pass, const rhi::ShaderProgram& program, const char* name,
                const glm::vec3& value) {
            const int32_t index = pass.GetPushConstant(program, name);
            if (index < 0) {
                return;
            }
            const uint32_t size = pass.GetPushConstantSize(index);
            if (size == sizeof(glm::vec3)) {
                pass.SetPushConstant(index, &value, size);
            } else if (size == sizeof(glm::vec4)) {
                const glm::vec4 wide(value, 0.0f);
                pass.SetPushConstant(index, &wide, size);
            } else {
                WarnOnce(std::string("DrawModel: push constant '") + name + "' is not a vec3/vec4; skipped");
            }
        }

        void FeedVec4(PassContext& pass, const rhi::ShaderProgram& program, const char* name,
                const glm::vec4& value) {
            const int32_t index = pass.GetPushConstant(program, name);
            if (index < 0) {
                return;
            }
            const uint32_t size = pass.GetPushConstantSize(index);
            if (size == sizeof(glm::vec4)) {
                pass.SetPushConstant(index, &value, size);
            } else if (size == sizeof(glm::vec3)) {
                const glm::vec3 narrow(value);
                pass.SetPushConstant(index, &narrow, size);
            } else {
                WarnOnce(std::string("DrawModel: push constant '") + name + "' is not a vec4/vec3; skipped");
            }
        }

        void FeedFloat(PassContext& pass, const rhi::ShaderProgram& program, const char* name,
                float value) {
            const int32_t index = pass.GetPushConstant(program, name);
            if (index < 0) {
                return;
            }
            if (pass.GetPushConstantSize(index) != sizeof(float)) {
                WarnOnce(std::string("DrawModel: push constant '") + name + "' is not a float; skipped");
                return;
            }
            pass.SetPushConstant(index, &value, sizeof(value));
        }

        void BindStandardTextures(PassContext& pass, const rhi::ShaderProgram& program,
                const Material& material, const ModelData& model, Assets& assets) {
            struct Entry {
                const char* mName;
                int32_t mTextureIndex;
            };
            const Entry entries[4] = {
                    {"baseColor", material.mBaseColorTexture},
                    {"metallicRoughness", material.mMetallicRoughnessTexture},
                    {"normal", material.mNormalTexture},
                    {"emissive", material.mEmissiveTexture},
            };
            for (const Entry& entry : entries) {
                if (entry.mTextureIndex < 0
                        || static_cast<size_t>(entry.mTextureIndex) >= model.mTextures.size()) {
                    continue;
                }
                UploadedTexture* texture = assets.GetTexture(model.mTextures[entry.mTextureIndex]);
                if (texture != nullptr) {
                    BindNamedTexture(pass, program, entry.mName, *texture);
                }
            }
        }

        void FeedMaterialFactors(PassContext& pass, const rhi::ShaderProgram& program,
                const Material& material) {
            FeedVec4(pass, program, "baseColorFactor", material.mBaseColor);
            FeedFloat(pass, program, "metallicFactor", material.mMetallic);
            FeedFloat(pass, program, "roughnessFactor", material.mRoughness);
            FeedVec4(pass, program, "emissiveFactor", material.mEmissiveColor);
        }

        void ApplyMaterialValues(PassContext& pass, const rhi::ShaderProgram& program,
                const MaterialSlot& slot, Assets& assets) {
            for (const MaterialValue& value : slot.mValues) {
                if (value.mType == MaterialValue::Type::kTexture) {
                    UploadedTexture* texture = assets.GetTexture(value.mTexture);
                    if (texture == nullptr) {
                        WarnOnce("DrawModel: material texture '" + value.mName + "' is invalid");
                        continue;
                    }
                    BindNamedTexture(pass, program, value.mName.c_str(), *texture);
                    continue;
                }

                const int32_t index = pass.GetPushConstant(program, value.mName.c_str());
                if (index < 0) {
                    DebugOnce("DrawModel: material value '" + value.mName + "' not found in program");
                    continue;
                }
                const void* data = nullptr;
                uint32_t size = 0;
                switch (value.mType) {
                    case MaterialValue::Type::kFloat:
                        data = &value.mFloat;
                        size = sizeof(float);
                        break;
                    case MaterialValue::Type::kVec2:
                        data = &value.mVec2;
                        size = sizeof(glm::vec2);
                        break;
                    case MaterialValue::Type::kVec3:
                        data = &value.mVec3;
                        size = sizeof(glm::vec3);
                        break;
                    case MaterialValue::Type::kVec4:
                        data = &value.mVec4;
                        size = sizeof(glm::vec4);
                        break;
                    case MaterialValue::Type::kInt:
                        data = &value.mInt;
                        size = sizeof(int32_t);
                        break;
                    case MaterialValue::Type::kMat4:
                        data = &value.mMat4;
                        size = sizeof(glm::mat4);
                        break;
                    case MaterialValue::Type::kTexture:
                        break;
                }
                if (size == 0) {
                    continue;
                }
                if (size != pass.GetPushConstantSize(index)) {
                    WarnOnce("DrawModel: material value '" + value.mName + "' size mismatch");
                    continue;
                }
                pass.SetPushConstant(index, data, size);
            }
        }

        void DrawModelImpl(PassContext& pass, const Camera* camera, Assets* assets,
                ModelHandle modelHandle, ProgramHandle defaultProgram, const glm::mat4& transform,
                bool forceProgram, const char* materialFilter) {
            if (assets == nullptr) {
                return;
            }
            ModelData* model = assets->GetModel(modelHandle);
            if (model == nullptr) {
                return;
            }

            for (const ModelDraw& draw : model->mDraws) {
                ModelPart* part = model->mParts.Get(draw.mPartIndex);
                if (part == nullptr) {
                    continue;
                }
                const MaterialSlot* slot = nullptr;
                if (part->mMaterialIndex != kInvalidIndex
                        && part->mMaterialIndex < model->mMaterials.size()) {
                    slot = &model->mMaterials[part->mMaterialIndex];
                }
                if (materialFilter != nullptr
                        && (slot == nullptr || slot->mMaterial.mName != materialFilter)) {
                    continue;
                }

                const rhi::ShaderProgram* program = nullptr;
                if (!forceProgram && slot != nullptr && slot->mProgram.IsValid()) {
                    program = assets->GetProgram(slot->mProgram);
                }
                if (program == nullptr) {
                    program = assets->GetProgram(defaultProgram);
                }
                if (program == nullptr) {
                    WarnOnce("DrawModel: no program (invalid default and no material program)");
                    continue;
                }

                pass.ClearTextureBindings();
                const glm::mat4 world = transform * draw.mWorld;
                if (slot != nullptr) {
                    BindStandardTextures(pass, *program, slot->mMaterial, *model, *assets);
                }
                FeedMat4(pass, *program, "model", world);
                if (camera != nullptr) {
                    const glm::mat4 viewProj = camera->mProj * camera->mView;
                    FeedMat4(pass, *program, "viewProj", viewProj);
                    FeedMat4(pass, *program, "view", camera->mView);
                    FeedVec3(pass, *program, "cameraPos", glm::vec3(glm::inverse(camera->mView)[3]));
                    FeedMat4(pass, *program, "mvp", viewProj * world);
                }
                if (slot != nullptr) {
                    FeedMaterialFactors(pass, *program, slot->mMaterial);
                    ApplyMaterialValues(pass, *program, *slot, *assets);
                }
                pass.Draw(part->mMesh, *program);
            }
        }
    }// namespace

    void PassContext::DrawModel(const Model& model, ProgramHandle defaultProgram,
            const glm::mat4& transform) {
        DrawModelImpl(*this, mRenderer->GetCameraInternal(), model.mAssets, model.mHandle,
                defaultProgram, transform, false, nullptr);
    }

    void PassContext::DrawModelForced(const Model& model, ProgramHandle program,
            const glm::mat4& transform) {
        DrawModelImpl(*this, mRenderer->GetCameraInternal(), model.mAssets, model.mHandle,
                program, transform, true, nullptr);
    }

    void PassContext::DrawModelPart(const Model& model, const char* materialName,
            ProgramHandle program, const glm::mat4& transform) {
        DrawModelImpl(*this, mRenderer->GetCameraInternal(), model.mAssets, model.mHandle,
                program, transform, true, materialName);
    }
}// namespace moe::neo
