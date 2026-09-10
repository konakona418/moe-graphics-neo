#include "Neo/Animation.hpp"

#include <algorithm>

namespace moe::neo {
    float AnimationClip::GetDuration() const {
        float duration = 0.0f;
        for (const auto& sampler : mSamplers) {
            if (!sampler.mInput.empty()) {
                duration = std::max(duration, sampler.mInput.back());
            }
        }
        return duration;
    }
}// namespace moe::neo