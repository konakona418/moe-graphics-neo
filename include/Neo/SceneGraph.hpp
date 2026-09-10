#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Core/SmallVector.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace moe::neo {
    // A scene-graph node. TRS transform with explicit parent/child links.
    // Nodes may be empty (no meshes) and must be preserved (glTF allows them).
    struct Node {
        std::string mName;
        glm::vec3 mTranslation{0.0f};
        glm::quat mRotation{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 mScale{1.0f};

        int32_t mParent{-1};
        moe::SmallVector<uint32_t, 4> mChildren;
        moe::SmallVector<uint32_t, 4> mMeshes; // indices into the owning Scene's meshes

        glm::mat4 LocalTransform() const;
    };

    // Ordered set of nodes. Node indices are stable (empty nodes included).
    struct SceneGraph {
        std::string mName;
        std::vector<Node> mNodes;
        moe::SmallVector<uint32_t, 8> mRootNodes;

        // Computes world transforms for all nodes (parents before children).
        std::vector<glm::mat4> ComputeWorldTransforms() const;
    };
}// namespace moe::neo