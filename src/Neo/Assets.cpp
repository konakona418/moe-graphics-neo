#include "Neo/Assets.hpp"
#include <Core/Profile.hpp>

#include "Neo/Importer.hpp"
#include "Neo/TextureLoader.hpp"

#include <Core/Error.hpp>
#include <Core/FileIo.hpp>
#include <Core/Logger.hpp>
#include <RHI/Device.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace moe::neo {
    namespace {
        // Part range of one scene mesh inside ModelData::mPartIndices.
        struct MeshRange {
            uint32_t mFirst{0};
            uint32_t mCount{0};
        };

        void AppendNodeDraws(const SceneGraph& graph, const std::vector<glm::mat4>& worlds,
                const SmallVector<MeshRange, 8>& meshRanges, ModelData& model, uint32_t nodeIndex) {
            const Node& node = graph.mNodes[nodeIndex];
            for (const uint32_t meshIndex : node.mMeshes) {
                if (meshIndex >= meshRanges.size()) {
                    continue;
                }
                const MeshRange& range = meshRanges[meshIndex];
                for (uint32_t i = 0; i < range.mCount; ++i) {
                    ModelDraw draw;
                    draw.mPartIndex = model.mPartIndices[range.mFirst + i];
                    draw.mWorld = worlds[nodeIndex];
                    model.mDraws.push_back(draw);
                }
            }
            for (const uint32_t child : node.mChildren) {
                AppendNodeDraws(graph, worlds, meshRanges, model, child);
            }
        }

        void UpsertValue(MaterialSlot& slot, MaterialValue value) {
            for (MaterialValue& existing : slot.mValues) {
                if (existing.mName == value.mName) {
                    existing = std::move(value);
                    return;
                }
            }
            slot.mValues.push_back(std::move(value));
        }
    }// namespace

    void ModelData::DestroyParts() {
        for (const uint32_t index : mPartIndices) {
            ModelPart* part = mParts.Get(index);
            if (part != nullptr) {
                part->mMesh.Destroy();
                mParts.Destruct(index);
            }
        }
        mPartIndices.clear();
    }

    int32_t ModelData::FindMaterial(const char* name) const {
        for (size_t i = 0; i < mMaterials.size(); ++i) {
            if (mMaterials[i].mMaterial.mName == name) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    }

    // ---- Model ----

    void Model::SetMaterialProgram(const char* materialName, ProgramHandle program) {
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialProgram(*data, materialName, program);
        }
    }

    void Model::SetMaterialTexture(const char* materialName, const char* name, TextureHandle texture) {
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialTexture(*data, materialName, name, texture);
        }
    }

    void Model::SetMaterialParam(const char* materialName, const char* name, float value) {
        MaterialValue materialValue;
        materialValue.mName = name;
        materialValue.mType = MaterialValue::Type::kFloat;
        materialValue.mFloat = value;
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialParam(*data, materialName, materialValue);
        }
    }

    void Model::SetMaterialParam(const char* materialName, const char* name, const glm::vec2& value) {
        MaterialValue materialValue;
        materialValue.mName = name;
        materialValue.mType = MaterialValue::Type::kVec2;
        materialValue.mVec2 = value;
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialParam(*data, materialName, materialValue);
        }
    }

    void Model::SetMaterialParam(const char* materialName, const char* name, const glm::vec3& value) {
        MaterialValue materialValue;
        materialValue.mName = name;
        materialValue.mType = MaterialValue::Type::kVec3;
        materialValue.mVec3 = value;
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialParam(*data, materialName, materialValue);
        }
    }

    void Model::SetMaterialParam(const char* materialName, const char* name, const glm::vec4& value) {
        MaterialValue materialValue;
        materialValue.mName = name;
        materialValue.mType = MaterialValue::Type::kVec4;
        materialValue.mVec4 = value;
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialParam(*data, materialName, materialValue);
        }
    }

    void Model::SetMaterialParam(const char* materialName, const char* name, int32_t value) {
        MaterialValue materialValue;
        materialValue.mName = name;
        materialValue.mType = MaterialValue::Type::kInt;
        materialValue.mInt = value;
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialParam(*data, materialName, materialValue);
        }
    }

    void Model::SetMaterialParam(const char* materialName, const char* name, const glm::mat4& value) {
        MaterialValue materialValue;
        materialValue.mName = name;
        materialValue.mType = MaterialValue::Type::kMat4;
        materialValue.mMat4 = value;
        ModelData* data = mAssets != nullptr ? mAssets->GetModel(mHandle) : nullptr;
        if (data != nullptr) {
            mAssets->SetMaterialParam(*data, materialName, materialValue);
        }
    }

    // ---- Assets ----

    Assets::~Assets() {
        Destroy();
    }

    bool Assets::Init(rhi::Device& device) {
        MOE_PROFILE_ZONE();
        if (mDevice != nullptr) {
            return moe::Fail("Assets already initialized");
        }
        if (!mUploader.Init(device)) {
            return false;
        }
        mDevice = &device;
        return true;
    }

    void Assets::Destroy() {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            return;
        }
        mModels.ForEach([](ModelData& model) { model.DestroyParts(); });
        mModels.Clear();
        mTextures.ForEach([](UploadedTexture& texture) { texture.Destroy(); });
        mTextures.Clear();
        mFonts.ForEach([](FontData& font) { font.Destroy(); });
        mFonts.Clear();
        mPrograms.Clear();
        mShaders.Clear();
        mDevice = nullptr;
        moe::Logger::Info("Assets destroyed");
    }

    Model Assets::LoadModel(const char* path) {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            moe::Error::Set("Assets: not initialized");
            return {};
        }
        Scene scene;
        if (!ImportGltf(path, scene)) {
            return {};
        }
        return UploadScene(scene);
    }

    Model Assets::UploadScene(const Scene& scene) {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            moe::Error::Set("Assets: not initialized");
            return {};
        }

        const ModelHandle handle = mModels.Emplace();
        ModelData* model = mModels.Get(handle);
        model->mName = scene.mName;

        auto rollback = [&] {
            for (const TextureHandle texture : model->mTextures) {
                UploadedTexture* uploaded = mTextures.Get(texture);
                if (uploaded != nullptr) {
                    uploaded->Destroy();
                    mTextures.Remove(texture);
                }
            }
            model->DestroyParts();
            mModels.Remove(handle);
        };

        // textures (parallel to Scene::mTextures; undecoded entries stay invalid)
        for (const Texture& texture : scene.mTextures) {
            if (texture.mData.empty()) {
                model->mTextures.push_back({});
                continue;
            }
            const TextureHandle textureHandle = UploadTexture(texture);
            if (!textureHandle.IsValid()) {
                rollback();
                return {};
            }
            model->mTextures.push_back(textureHandle);
        }

        // one uploaded mesh per primitive (material boundaries survive)
        SmallVector<MeshRange, 8> meshRanges;
        for (const Mesh& mesh : scene.mMeshes) {
            const uint32_t first = static_cast<uint32_t>(model->mPartIndices.size());
            for (const MeshPrimitive& primitive : mesh.mPrimitives) {
                const uint32_t partIndex = model->mParts.Acquire();
                ModelPart* part = model->mParts.Construct(partIndex);
                part->mMaterialIndex = primitive.mMaterialIndex >= 0
                        ? static_cast<uint32_t>(primitive.mMaterialIndex) : kInvalidIndex;
                model->mPartIndices.push_back(partIndex);
                if (!mUploader.UploadMeshPrimitive(primitive, part->mMesh)) {
                    rollback();
                    return {};
                }
            }
            meshRanges.push_back({first, static_cast<uint32_t>(model->mPartIndices.size()) - first});
        }

        // material slots (parallel to Scene::mMaterials)
        for (const Material& material : scene.mMaterials) {
            MaterialSlot slot;
            slot.mMaterial = material;
            model->mMaterials.push_back(std::move(slot));
        }

        // flattened draws with precomputed world transforms
        if (scene.mGraph.mNodes.empty()) {
            for (size_t meshIndex = 0; meshIndex < meshRanges.size(); ++meshIndex) {
                const MeshRange& range = meshRanges[meshIndex];
                for (uint32_t i = 0; i < range.mCount; ++i) {
                    ModelDraw draw;
                    draw.mPartIndex = model->mPartIndices[range.mFirst + i];
                    model->mDraws.push_back(draw);
                }
            }
        } else {
            const std::vector<glm::mat4> worlds = scene.mGraph.ComputeWorldTransforms();
            for (const uint32_t root : scene.mGraph.mRootNodes) {
                AppendNodeDraws(scene.mGraph, worlds, meshRanges, *model, root);
            }
        }

        moe::Logger::Info("Uploaded model '{}' ({} parts, {} textures, {} draws)",
                model->mName, model->mPartIndices.size(), model->mTextures.size(), model->mDraws.size());
        return Model(this, handle);
    }

    TextureHandle Assets::LoadTexture(const char* path, bool srgb) {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            moe::Error::Set("Assets: not initialized");
            return {};
        }
        std::vector<uint8_t> bytes;
        if (!moe::ReadFileBytes(path, bytes)) {
            return {};
        }
        Texture texture;
        if (!DecodeTexture(bytes.data(), bytes.size(), texture, srgb)) {
            moe::Error::Set(std::string("Assets: failed to decode texture: ") + path);
            return {};
        }
        texture.mName = path;
        return UploadTexture(texture);
    }

    TextureHandle Assets::UploadTexture(const Texture& texture) {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            moe::Error::Set("Assets: not initialized");
            return {};
        }
        const TextureHandle handle = mTextures.Emplace();
        if (!mUploader.UploadTexture(texture, *mTextures.Get(handle))) {
            mTextures.Remove(handle);
            return {};
        }
        return handle;
    }

    ProgramHandle Assets::LoadGraphicsProgram(const char* vertexPath, const char* fragmentPath) {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            moe::Error::Set("Assets: not initialized");
            return {};
        }
        const Handle<rhi::Shader> vertex = mShaders.Emplace();
        if (!mShaders.Get(vertex)->Load(vertexPath, rhi::ShaderStage::kVertex)) {
            mShaders.Remove(vertex);
            return {};
        }
        const Handle<rhi::Shader> fragment = mShaders.Emplace();
        if (!mShaders.Get(fragment)->Load(fragmentPath, rhi::ShaderStage::kFragment)) {
            mShaders.Remove(fragment);
            mShaders.Remove(vertex);
            return {};
        }
        const ProgramHandle handle = mPrograms.Emplace();
        rhi::ShaderProgram* program = mPrograms.Get(handle);
        if (!program->AddShader(*mShaders.Get(vertex)) || !program->AddShader(*mShaders.Get(fragment))) {
            moe::Error::Set("Assets: failed to build graphics program");
            mPrograms.Remove(handle);
            mShaders.Remove(fragment);
            mShaders.Remove(vertex);
            return {};
        }
        return handle;
    }

    ProgramHandle Assets::LoadComputeProgram(const char* path) {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            moe::Error::Set("Assets: not initialized");
            return {};
        }
        const Handle<rhi::Shader> shader = mShaders.Emplace();
        if (!mShaders.Get(shader)->Load(path, rhi::ShaderStage::kCompute)) {
            mShaders.Remove(shader);
            return {};
        }
        const ProgramHandle handle = mPrograms.Emplace();
        if (!mPrograms.Get(handle)->AddShader(*mShaders.Get(shader))) {
            moe::Error::Set("Assets: failed to build compute program");
            mPrograms.Remove(handle);
            mShaders.Remove(shader);
            return {};
        }
        return handle;
    }

    ModelData* Assets::GetModel(ModelHandle handle) {
        return mModels.Get(handle);
    }

    UploadedTexture* Assets::GetTexture(TextureHandle handle) {
        return mTextures.Get(handle);
    }

    rhi::ShaderProgram* Assets::GetProgram(ProgramHandle handle) {
        return mPrograms.Get(handle);
    }

    void Assets::SetMaterialProgram(ModelData& model, const char* materialName, ProgramHandle program) {
        const int32_t index = model.FindMaterial(materialName);
        if (index < 0) {
            moe::Error::Set(std::string("Assets: unknown material '") + materialName + "'");
            return;
        }
        model.mMaterials[static_cast<size_t>(index)].mProgram = program;
    }

    void Assets::SetMaterialTexture(ModelData& model, const char* materialName, const char* name,
            TextureHandle texture) {
        const int32_t index = model.FindMaterial(materialName);
        if (index < 0) {
            moe::Error::Set(std::string("Assets: unknown material '") + materialName + "'");
            return;
        }
        MaterialValue value;
        value.mName = name;
        value.mType = MaterialValue::Type::kTexture;
        value.mTexture = texture;
        UpsertValue(model.mMaterials[static_cast<size_t>(index)], std::move(value));
    }

    void Assets::SetMaterialParam(ModelData& model, const char* materialName, const MaterialValue& value) {
        const int32_t index = model.FindMaterial(materialName);
        if (index < 0) {
            moe::Error::Set(std::string("Assets: unknown material '") + materialName + "'");
            return;
        }
        UpsertValue(model.mMaterials[static_cast<size_t>(index)], value);
    }
}// namespace moe::neo
