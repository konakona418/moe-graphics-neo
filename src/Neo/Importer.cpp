#include "Neo/Importer.hpp"

#include <Neo/TextureLoader.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

namespace moe::neo {
    namespace {
        constexpr fastgltf::Options kOptions = fastgltf::Options::LoadExternalBuffers
                | fastgltf::Options::LoadExternalImages
                | fastgltf::Options::LoadGLBBuffers
                | fastgltf::Options::DecomposeNodeMatrices;

        // Extracts the raw bytes of a loaded fastgltf buffer (variant source).
        const uint8_t* BufferBytes(const fastgltf::Buffer& buffer, size_t* outSize) {
            const uint8_t* bytes = nullptr;
            bool ok = std::visit(
                    [&](const auto& source) -> bool {
                        using Source = std::remove_cvref_t<decltype(source)>;
                        if constexpr (std::is_same_v<Source, fastgltf::sources::Vector>
                                || std::is_same_v<Source, fastgltf::sources::Array>
                                || std::is_same_v<Source, fastgltf::sources::ByteView>) {
                            bytes = reinterpret_cast<const uint8_t*>(source.bytes.data());
                            *outSize = source.bytes.size();
                            return true;
                        } else {
                            return false;
                        }
                    },
                    buffer.data);
            return ok ? bytes : nullptr;
        }

        // Extracts the raw image bytes from a fastgltf image source. Returns
        // false for sources we don't handle (external URI files, custom).
        bool ImageBytes(const fastgltf::Asset& asset, const fastgltf::Image& image,
                const uint8_t** outBytes, size_t* outSize) {
            return std::visit(
                    [&](const auto& source) -> bool {
                        using Source = std::remove_cvref_t<decltype(source)>;
                        if constexpr (std::is_same_v<Source, fastgltf::sources::BufferView>) {
                            if (source.bufferViewIndex >= asset.bufferViews.size()) {
                                return false;
                            }
                            const fastgltf::BufferView& view = asset.bufferViews[source.bufferViewIndex];
                            if (view.bufferIndex >= asset.buffers.size()) {
                                return false;
                            }
                            size_t bufferSize = 0;
                            const uint8_t* bufferBytes = BufferBytes(asset.buffers[view.bufferIndex],
                                    &bufferSize);
                            if (bufferBytes == nullptr
                                    || view.byteOffset + view.byteLength > bufferSize) {
                                return false;
                            }
                            *outBytes = bufferBytes + view.byteOffset;
                            *outSize = view.byteLength;
                            return true;
                        } else if constexpr (std::is_same_v<Source, fastgltf::sources::Vector>
                                || std::is_same_v<Source, fastgltf::sources::Array>
                                || std::is_same_v<Source, fastgltf::sources::ByteView>) {
                            *outBytes = reinterpret_cast<const uint8_t*>(source.bytes.data());
                            *outSize = source.bytes.size();
                            return true;
                        } else {
                            return false; // URI (external) / custom / monostate
                        }
                    },
                    image.data);
        }

        glm::vec3 ToVec3(const fastgltf::math::fvec3& v) {
            return glm::vec3(v.x(), v.y(), v.z());
        }

        glm::quat ToQuat(const fastgltf::math::fquat& q) {
            return glm::quat(q.w(), q.x(), q.y(), q.z());
        }

        template<typename T>
        bool ReadAccessor(const fastgltf::Asset& asset, std::size_t accessorIndex,
                std::vector<T>& out, std::string& error) {
            if (accessorIndex >= asset.accessors.size()) {
                error = "glTF: accessor index out of range";
                return false;
            }
            const fastgltf::Accessor& accessor = asset.accessors[accessorIndex];
            out.clear();
            out.reserve(accessor.count);
            fastgltf::iterateAccessorWithIndex<T>(asset, accessor,
                    [&](T&& value, std::size_t) { out.push_back(std::move(value)); });
            return true;
        }
    }// namespace

    bool ImportGltf(const char* path, Scene& outScene, std::string& outError) {
        fastgltf::Parser parser;
        auto data = fastgltf::GltfDataBuffer::FromPath(path);
        if (data.error() != fastgltf::Error::None) {
            outError = "glTF: failed to read file";
            return false;
        }
        auto result = parser.loadGltf(data.get(), std::filesystem::path(path).parent_path(), kOptions);
        if (result.error() != fastgltf::Error::None) {
            outError = "glTF: parse failed";
            return false;
        }
        fastgltf::Asset asset = std::move(result.get());

        Scene scene;
        scene.mName = std::filesystem::path(path).stem().string();

        // ---- nodes + scene graph (preserve empty nodes, full hierarchy) ----
        scene.mGraph.mNodes.resize(asset.nodes.size());
        for (std::size_t i = 0; i < asset.nodes.size(); ++i) {
            const fastgltf::Node& node = asset.nodes[i];
            Node& outNode = scene.mGraph.mNodes[i];
            outNode.mName = node.name;
            if (std::holds_alternative<fastgltf::TRS>(node.transform)) {
                const fastgltf::TRS& trs = std::get<fastgltf::TRS>(node.transform);
                outNode.mTranslation = ToVec3(trs.translation);
                outNode.mRotation = ToQuat(trs.rotation);
                outNode.mScale = ToVec3(trs.scale);
            }
            if (node.meshIndex.has_value()) {
                outNode.mMeshes.push_back(static_cast<uint32_t>(*node.meshIndex));
            }
            for (const std::size_t child : node.children) {
                outNode.mChildren.push_back(static_cast<uint32_t>(child));
                scene.mGraph.mNodes[child].mParent = static_cast<int32_t>(i);
            }
        }
        for (std::size_t i = 0; i < scene.mGraph.mNodes.size(); ++i) {
            if (scene.mGraph.mNodes[i].mParent < 0) {
                scene.mGraph.mRootNodes.push_back(static_cast<uint32_t>(i));
            }
        }

        // ---- default scene roots ----
        if (asset.defaultScene.has_value() && *asset.defaultScene < asset.scenes.size()) {
            for (const std::size_t nodeIndex : asset.scenes[*asset.defaultScene].nodeIndices) {
                scene.mSceneRoots.push_back(static_cast<uint32_t>(nodeIndex));
            }
        } else {
            scene.mSceneRoots = scene.mGraph.mRootNodes;
        }

        // ---- meshes ----
        scene.mMeshes.resize(asset.meshes.size());
        for (std::size_t m = 0; m < asset.meshes.size(); ++m) {
            const fastgltf::Mesh& mesh = asset.meshes[m];
            Mesh& outMesh = scene.mMeshes[m];
            outMesh.mName = mesh.name;
            outMesh.mPrimitives.resize(mesh.primitives.size());
            for (std::size_t p = 0; p < mesh.primitives.size(); ++p) {
                const fastgltf::Primitive& prim = mesh.primitives[p];
                MeshPrimitive& outPrim = outMesh.mPrimitives[p];
                std::string ignored;

                const auto* pos = prim.findAttribute("POSITION");
                if (pos != nullptr) {
                    ReadAccessor<glm::vec3>(asset, pos->accessorIndex, outPrim.mPositions, ignored);
                    for (const glm::vec3& v : outPrim.mPositions) {
                        outPrim.mMin = glm::min(outPrim.mMin, v);
                        outPrim.mMax = glm::max(outPrim.mMax, v);
                    }
                }
                const auto* nrm = prim.findAttribute("NORMAL");
                if (nrm != nullptr) {
                    ReadAccessor<glm::vec3>(asset, nrm->accessorIndex, outPrim.mNormals, ignored);
                }
                const auto* uv = prim.findAttribute("TEXCOORD_0");
                if (uv != nullptr) {
                    ReadAccessor<glm::vec2>(asset, uv->accessorIndex, outPrim.mUv0, ignored);
                }
                const auto* tan = prim.findAttribute("TANGENT");
                if (tan != nullptr) {
                    ReadAccessor<glm::vec4>(asset, tan->accessorIndex, outPrim.mTangents, ignored);
                }
                const auto* joints = prim.findAttribute("JOINTS_0");
                if (joints != nullptr) {
                    ReadAccessor<glm::uvec4>(asset, joints->accessorIndex, outPrim.mJointIndices, ignored);
                }
                const auto* weights = prim.findAttribute("WEIGHTS_0");
                if (weights != nullptr) {
                    ReadAccessor<glm::vec4>(asset, weights->accessorIndex, outPrim.mJointWeights, ignored);
                }
                if (prim.indicesAccessor.has_value()) {
                    ReadAccessor<uint32_t>(asset, *prim.indicesAccessor, outPrim.mIndices, ignored);
                }
                outPrim.mMaterialIndex = prim.materialIndex.has_value()
                        ? static_cast<int32_t>(*prim.materialIndex) : -1;
            }
        }

        // ---- materials ----
        scene.mMaterials.resize(asset.materials.size());
        for (std::size_t i = 0; i < asset.materials.size(); ++i) {
            const fastgltf::Material& mat = asset.materials[i];
            Material& out = scene.mMaterials[i];
            out.mName = mat.name;
            out.mBaseColor = glm::vec4(
                    mat.pbrData.baseColorFactor.x(), mat.pbrData.baseColorFactor.y(),
                    mat.pbrData.baseColorFactor.z(), mat.pbrData.baseColorFactor.w());
            out.mMetallic = mat.pbrData.metallicFactor;
            out.mRoughness = mat.pbrData.roughnessFactor;
            out.mEmissiveColor = glm::vec4(
                    mat.emissiveFactor.x(), mat.emissiveFactor.y(), mat.emissiveFactor.z(), 1.0f);
            out.mDoubleSided = mat.doubleSided;
            if (mat.pbrData.baseColorTexture.has_value()) {
                out.mBaseColorTexture = static_cast<int32_t>(mat.pbrData.baseColorTexture->textureIndex);
            }
            if (mat.pbrData.metallicRoughnessTexture.has_value()) {
                out.mMetallicRoughnessTexture = static_cast<int32_t>(mat.pbrData.metallicRoughnessTexture->textureIndex);
            }
            if (mat.normalTexture.has_value()) {
                out.mNormalTexture = static_cast<int32_t>(mat.normalTexture->textureIndex);
            }
            if (mat.emissiveTexture.has_value()) {
                out.mEmissiveTexture = static_cast<int32_t>(mat.emissiveTexture->textureIndex);
            }
        }

        // ---- textures (decode pixels; material references already point at
        // the texture indices) ----
        scene.mTextures.resize(asset.textures.size());
        for (std::size_t i = 0; i < asset.textures.size(); ++i) {
            const fastgltf::Texture& tex = asset.textures[i];
            Texture& out = scene.mTextures[i];
            if (!tex.imageIndex.has_value() || *tex.imageIndex >= asset.images.size()) {
                continue;
            }
            const fastgltf::Image& image = asset.images[*tex.imageIndex];
            out.mName = image.name;

            const uint8_t* bytes = nullptr;
            size_t size = 0;
            if (!ImageBytes(asset, image, &bytes, &size)
                    || !DecodeTexture(bytes, size, out, false)) {
                out.mChannels = 0; // undecoded
            }
        }

        // sRGB semantics: base color and emissive textures are color data.
        for (const Material& material : scene.mMaterials) {
            for (const int32_t slot : {material.mBaseColorTexture, material.mEmissiveTexture}) {
                if (slot != kNoTexture
                        && slot >= 0 && slot < static_cast<int32_t>(scene.mTextures.size())) {
                    scene.mTextures[static_cast<size_t>(slot)].mSrgb = true;
                }
            }
        }

        // ---- skeletons ----
        scene.mSkeletons.resize(asset.skins.size());
        for (std::size_t i = 0; i < asset.skins.size(); ++i) {
            const fastgltf::Skin& skin = asset.skins[i];
            Skeleton& out = scene.mSkeletons[i];
            out.mName = skin.name;
            for (const std::size_t joint : skin.joints) {
                out.mJointNodes.push_back(static_cast<uint32_t>(joint));
            }
            if (skin.inverseBindMatrices.has_value()) {
                std::string ignored;
                ReadAccessor<glm::mat4>(asset, *skin.inverseBindMatrices, out.mInverseBindMatrices, ignored);
            }
        }
        // mark skinned meshes (a node with a skin + a mesh skins that mesh)
        for (std::size_t i = 0; i < asset.nodes.size(); ++i) {
            const fastgltf::Node& node = asset.nodes[i];
            if (node.skinIndex.has_value() && node.meshIndex.has_value()) {
                Mesh& mesh = scene.mMeshes[*node.meshIndex];
                mesh.mSkinIndex = static_cast<int32_t>(*node.skinIndex);
            }
        }

        // ---- animations ----
        scene.mAnimations.resize(asset.animations.size());
        for (std::size_t a = 0; a < asset.animations.size(); ++a) {
            const fastgltf::Animation& anim = asset.animations[a];
            AnimationClip& out = scene.mAnimations[a];
            out.mName = anim.name;
            out.mSamplers.resize(anim.samplers.size());
            for (std::size_t s = 0; s < anim.samplers.size(); ++s) {
                const fastgltf::AnimationSampler& sampler = anim.samplers[s];
                AnimationSampler& outSampler = out.mSamplers[s];
                std::string ignored;
                ReadAccessor<float>(asset, sampler.inputAccessor, outSampler.mInput, ignored);
                ReadAccessor<glm::vec4>(asset, sampler.outputAccessor, outSampler.mOutput, ignored);
                switch (sampler.interpolation) {
                    case fastgltf::AnimationInterpolation::Step: outSampler.mInterpolation = AnimationSampler::Interpolation::kStep; break;
                    case fastgltf::AnimationInterpolation::CubicSpline: outSampler.mInterpolation = AnimationSampler::Interpolation::kCubicSpline; break;
                    default: outSampler.mInterpolation = AnimationSampler::Interpolation::kLinear; break;
                }
            }
            for (const fastgltf::AnimationChannel& channel : anim.channels) {
                AnimationChannel& outChannel = out.mChannels.emplace_back();
                outChannel.mSamplerIndex = static_cast<uint32_t>(channel.samplerIndex);
                outChannel.mTargetNode = channel.nodeIndex.has_value()
                        ? static_cast<uint32_t>(*channel.nodeIndex) : 0;
                switch (channel.path) {
                    case fastgltf::AnimationPath::Rotation: outChannel.mTarget = AnimationChannel::Target::kRotation; break;
                    case fastgltf::AnimationPath::Scale: outChannel.mTarget = AnimationChannel::Target::kScale; break;
                    case fastgltf::AnimationPath::Weights: outChannel.mTarget = AnimationChannel::Target::kWeights; break;
                    default: outChannel.mTarget = AnimationChannel::Target::kTranslation; break;
                }
            }
        }

        outScene = std::move(scene);
        return true;
    }
}// namespace moe::neo