#pragma once

#include <cstdint>
#include <string>

#include <glm/glm.hpp>

namespace moe::neo {
    // PBR material. Texture slots are indices into the owning Scene's textures
    // (kNoTexture = unbound).
    constexpr int32_t kNoTexture = -1;

    struct Material {
        std::string mName;
        glm::vec4 mBaseColor{1.0f};
        float mMetallic{0.0f};
        float mRoughness{1.0f};
        glm::vec4 mEmissiveColor{0.0f};
        float mEmissiveStrength{1.0f};

        int32_t mBaseColorTexture{kNoTexture};
        int32_t mMetallicRoughnessTexture{kNoTexture};
        int32_t mNormalTexture{kNoTexture};
        int32_t mEmissiveTexture{kNoTexture};

        bool mDoubleSided{false};
    };
}// namespace moe::neo