#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace moe::neo {
    // CPU-side mesh. Attributes are stored as separate arrays (flexible for
    // editing, e.g. snow deformation); the GPU uploader decides how to lay them
    // out in vertex buffers.
    struct MeshPrimitive {
        std::vector<glm::vec3> mPositions;
        std::vector<glm::vec3> mNormals;
        std::vector<glm::vec2> mUv0;
        std::vector<glm::vec4> mColors;        // rgba; uploaded as RGBA8
        std::vector<glm::vec4> mTangents;      // xyz = tangent, w = sign
        std::vector<glm::uvec4> mJointIndices; // skinning
        std::vector<glm::vec4> mJointWeights;  // skinning
        std::vector<uint32_t> mIndices;

        int32_t mMaterialIndex{-1};
        glm::vec3 mMin{0.0f};
        glm::vec3 mMax{0.0f};
    };

    struct Mesh {
        std::string mName;
        std::vector<MeshPrimitive> mPrimitives;

        // skin index into the owning Scene's skeletons; -1 = not skinned
        int32_t mSkinIndex{-1};
    };
}// namespace moe::neo