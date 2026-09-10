#pragma once

#include "Neo/Cache.hpp"
#include "Neo/Font.hpp"
#include "Neo/Material.hpp"
#include "Neo/Scene.hpp"
#include "Neo/Texture.hpp"
#include "Neo/Uploader.hpp"

#include <RHI/Shader.hpp>

#include <Core/Pool.hpp>
#include <Core/SmallVector.hpp>

#include <glm/glm.hpp>

#include <cstdint>
#include <string>

namespace moe::neo {
    class Assets;

    // Camera: pure data. DrawModel feeds viewProj/view/cameraPos to shaders
    // that declare those names; shaders that want anything else (inverse
    // matrices, custom projections) compute and push it themselves.
    struct Camera {
        glm::mat4 mView{1.0f};
        glm::mat4 mProj{1.0f};
    };

    using ProgramHandle = Handle<rhi::ShaderProgram>;
    using TextureHandle = Handle<UploadedTexture>;

    // One per-material override: a named texture or a named push-constant
    // value. Resolved against the drawing program's reflection at draw time
    // (unresolved names are skipped with a one-time warning).
    struct MaterialValue {
        enum class Type { kFloat, kVec2, kVec3, kVec4, kInt, kMat4, kTexture };

        std::string mName;
        Type mType{Type::kFloat};
        float mFloat{0.0f};
        glm::vec2 mVec2{0.0f};
        glm::vec3 mVec3{0.0f};
        glm::vec4 mVec4{0.0f};
        int32_t mInt{0};
        glm::mat4 mMat4{1.0f};
        TextureHandle mTexture;
    };

    // Per-material runtime state: the imported CPU material, the program that
    // draws it (invalid = fall back to DrawModel's default), and named
    // overrides for that program's private parameters.
    struct MaterialSlot {
        Material mMaterial;
        ProgramHandle mProgram;
        SmallVector<MaterialValue, 8> mValues;
    };

    struct ModelPart {
        UploadedMesh mMesh;
        uint32_t mMaterialIndex{kInvalidIndex};
    };

    // One drawable primitive with its precomputed world transform.
    struct ModelDraw {
        uint32_t mPartIndex{0}; // into ModelData::mParts
        glm::mat4 mWorld{1.0f};
    };

    // GPU-side model: one uploaded mesh per primitive (material boundaries
    // survive; UploadMesh would concatenate them), texture handles, material
    // slots, and the flattened node traversal with world transforms.
    struct ModelData {
        std::string mName;
        Pool<ModelPart> mParts;
        SmallVector<uint32_t, 16> mPartIndices;
        SmallVector<TextureHandle, 8> mTextures; // parallel to Scene::mTextures
        SmallVector<MaterialSlot, 8> mMaterials;
        SmallVector<ModelDraw, 16> mDraws;

        void DestroyParts();
        int32_t FindMaterial(const char* name) const;
    };

    using ModelHandle = Handle<ModelData>;

    // Lightweight value handle to a model. Configuration resolves through the
    // owning Assets; stale handles are inert.
    class Model {
    public:
        Model() = default;

        bool IsValid() const {
            return mAssets != nullptr && mHandle.IsValid();
        }

        void SetMaterialProgram(const char* materialName, ProgramHandle program);
        void SetMaterialTexture(const char* materialName, const char* name, TextureHandle texture);
        void SetMaterialParam(const char* materialName, const char* name, float value);
        void SetMaterialParam(const char* materialName, const char* name, const glm::vec2& value);
        void SetMaterialParam(const char* materialName, const char* name, const glm::vec3& value);
        void SetMaterialParam(const char* materialName, const char* name, const glm::vec4& value);
        void SetMaterialParam(const char* materialName, const char* name, int32_t value);
        void SetMaterialParam(const char* materialName, const char* name, const glm::mat4& value);

    private:
        friend class Assets;
        friend class PassContext;

        Model(Assets* assets, ModelHandle handle)
            : mAssets(assets)
            , mHandle(handle) {}

        Assets* mAssets{nullptr};
        ModelHandle mHandle;
    };

    // Content layer: owns the uploader and every GPU asset, addressed by
    // generation-checked handles. Resources live until Destroy() (pool
    // semantics: no per-resource release in the normal path).
    class Assets {
    public:
        Assets() = default;
        ~Assets();

        Assets(const Assets&) = delete;
        Assets& operator=(const Assets&) = delete;

        bool Init(rhi::Device& device);
        void Destroy();

        // glTF import + upload in one call.
        Model LoadModel(const char* path);
        // Uploads an already-built scene (procedural geometry goes here).
        Model UploadScene(const Scene& scene);

        TextureHandle LoadTexture(const char* path, bool srgb);
        TextureHandle UploadTexture(const Texture& texture);

        // Loads a TrueType font and preprocesses the glyphs for the text
        // renderer: ASCII plus every codepoint in `sampleText` is extracted
        // (outline curves, band acceleration structure, kerning).
        Font LoadFont(const char* path, std::string_view sampleText);

        ProgramHandle LoadGraphicsProgram(const char* vertexPath, const char* fragmentPath);
        ProgramHandle LoadComputeProgram(const char* path);

        ModelData* GetModel(ModelHandle handle);
        UploadedTexture* GetTexture(TextureHandle handle);
        FontData* GetFont(FontHandle handle);
        rhi::ShaderProgram* GetProgram(ProgramHandle handle);

    private:
        friend class Model;

        void SetMaterialProgram(ModelData& model, const char* materialName, ProgramHandle program);
        void SetMaterialTexture(ModelData& model, const char* materialName, const char* name,
                TextureHandle texture);
        void SetMaterialParam(ModelData& model, const char* materialName, const MaterialValue& value);

        rhi::Device* mDevice{nullptr};
        Uploader mUploader;
        Cache<ModelData> mModels;
        Cache<UploadedTexture> mTextures;
        Cache<FontData> mFonts;
        Cache<rhi::ShaderProgram> mPrograms;
        Cache<rhi::Shader> mShaders;
    };
}// namespace moe::neo
