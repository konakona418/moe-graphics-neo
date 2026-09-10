#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace moe::neo {
    // Animation is format-agnostic: a clip is a set of channels whose keyframes
    // live in samplers, each targeting a scene node's TRS (or weights). FBX and
    // glTF both feed this model. Runtime blending (pose stacks) is a separate,
    // later concern.
    struct AnimationSampler {
        enum class Interpolation : uint8_t {
            kLinear,
            kStep,
            kCubicSpline,
        };

        Interpolation mInterpolation{Interpolation::kLinear};
        std::vector<float> mInput;          // keyframe times
        std::vector<glm::vec4> mOutput;     // keyframe values (vec4 covers vec3/quat)
    };

    struct AnimationChannel {
        enum class Target : uint8_t {
            kTranslation,
            kRotation,
            kScale,
            kWeights,
        };

        Target mTarget{Target::kTranslation};
        uint32_t mTargetNode{0};  // node index in the scene graph
        uint32_t mSamplerIndex{0};
    };

    struct AnimationClip {
        std::string mName;
        std::vector<AnimationSampler> mSamplers;
        std::vector<AnimationChannel> mChannels;

        float GetDuration() const; // max input time across samplers
    };
}// namespace moe::neo