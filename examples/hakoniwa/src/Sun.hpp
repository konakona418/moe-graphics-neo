#pragma once

#include <glm/glm.hpp>

#include <cmath>

// Shared sun helpers (direction + colour ramps), used by the app shell, the
// sky pass and the HUD.

namespace hakoniwa {
    inline glm::vec3 MakeSunDirection(float elevationDeg, float azimuthDeg) {
        const float e = glm::radians(elevationDeg);
        const float a = glm::radians(azimuthDeg);
        return glm::normalize(
                glm::vec3(glm::cos(e) * glm::sin(a), glm::sin(e), glm::cos(e) * glm::cos(a)));
    }

    inline glm::vec3 SunColorFor(const glm::vec3& sunDir) {
        const float t = glm::clamp((sunDir.y + 0.05f) / 0.7f, 0.0f, 1.0f);
        const glm::vec3 low(1.0f, 0.45f, 0.18f);
        const glm::vec3 mid(1.0f, 0.72f, 0.40f);
        const glm::vec3 high(1.0f, 0.97f, 0.92f);
        return t < 0.5f ? glm::mix(low, mid, t * 2.0f) : glm::mix(mid, high, (t - 0.5f) * 2.0f);
    }

    inline glm::vec3 SkyHorizonFor(const glm::vec3& sunDir) {
        const float dusk = 1.0f - glm::smoothstep(0.02f, 0.5f, sunDir.y);
        const glm::vec3 day(0.62f, 0.78f, 0.95f);
        const glm::vec3 duskHorizon(1.0f, 0.50f, 0.26f);
        return glm::mix(day, duskHorizon, dusk);
    }
}// namespace hakoniwa
