#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace moe::neo {
    // Joint hierarchy + inverse bind matrices for GPU skinning. The hierarchy
    // is expressed through scene node indices (mJointNodes).
    struct Skeleton {
        std::string mName;
        std::vector<uint32_t> mJointNodes;           // node indices in the scene graph
        std::vector<glm::mat4> mInverseBindMatrices; // per joint
    };
}// namespace moe::neo