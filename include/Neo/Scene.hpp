#pragma once

#include "Neo/Animation.hpp"
#include "Neo/Material.hpp"
#include "Neo/Mesh.hpp"
#include "Neo/SceneGraph.hpp"
#include "Neo/Skeleton.hpp"
#include "Neo/Texture.hpp"

#include <Core/SmallVector.hpp>

#include <string>
#include <vector>

namespace moe::neo {
    // A fully imported asset: format-agnostic CPU data produced by an importer
    // (glTF today, FBX later). Everything lives here; the GPU uploader copies
    // what is needed and the CPU data can be discarded.
    struct Scene {
        std::string mName;
        SceneGraph mGraph;
        std::vector<Mesh> mMeshes;
        std::vector<Material> mMaterials;
        std::vector<Texture> mTextures;
        std::vector<Skeleton> mSkeletons;
        std::vector<AnimationClip> mAnimations;

        // Which of the graph's nodes are "default scene" roots (glTF scene).
        moe::SmallVector<uint32_t, 8> mSceneRoots;
    };
}// namespace moe::neo