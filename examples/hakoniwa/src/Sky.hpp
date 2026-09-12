#pragma once

#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>

#include <glm/glm.hpp>

#include <cstdint>

// Sky module: a fullscreen gradient sky + sun disc. Drawn first, clearing the
// scene colour + depth, so terrain and grass render over it. Authored in HDR so
// the shared post tone map handles it with the rest of the scene.

namespace hakoniwa {
    struct SkyFrame {
        glm::vec3 mForward{0.0f, 0.0f, -1.0f};
        glm::vec3 mRight{1.0f, 0.0f, 0.0f};
        glm::vec3 mUp{0.0f, 1.0f, 0.0f};
        float mTanHalfFov{1.0f};
        float mAspect{1.0f};
        glm::vec3 mSunDir{0.0f, 1.0f, 0.0f};
        glm::vec3 mSunColor{1.0f};
        glm::vec3 mHorizonColor{0.62f, 0.78f, 0.95f};
        glm::vec3 mZenithColor{0.16f, 0.34f, 0.66f};
    };

    class Sky {
    public:
        bool Init(moe::neo::Assets& assets, moe::neo::Renderer& renderer);
        void Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
                moe::neo::RenderTargetHandle target, const SkyFrame& frame);
        void Destroy();

    private:
        moe::neo::ProgramHandle mProgram;
        int32_t mPcForward{-1};
        int32_t mPcRight{-1};
        int32_t mPcUp{-1};
        int32_t mPcTanHalfFov{-1};
        int32_t mPcAspect{-1};
        int32_t mPcSunDir{-1};
        int32_t mPcSunColor{-1};
        int32_t mPcHorizonColor{-1};
        int32_t mPcZenithColor{-1};
    };
}// namespace hakoniwa
