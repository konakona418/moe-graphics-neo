#include "Physics.hpp"

#include <cmath>

namespace hakoniwa {
    namespace {
        constexpr float kGravity = -20.0f;
        constexpr float kJumpSpeed = 7.5f;
        constexpr float kWalkableMargin = 1.0f;

        // Keep in sync with Terrain.cpp's height lambda and grass.slang.
        float TerrainHeight(float x, float z, float amplitude) {
            return amplitude
                    * (0.5f * std::sin(x * 0.012f) * std::cos(z * 0.011f)
                            + 0.3f * std::sin(x * 0.03f + 1.7f)
                            + 0.2f * std::cos(z * 0.025f - 0.6f));
        }
    }// namespace

    struct Physics::Impl {
        float mTerrainSize{600.0f};
        float mTerrainAmplitude{20.0f};
        glm::vec3 mPosition{0.0f};
        glm::vec3 mVelocity{0.0f};
        bool mOnGround{false};
    };

    Physics::Physics() = default;
    Physics::~Physics() = default;

    bool Physics::Init(const PhysicsParams& params, const glm::vec3& playerPosition) {
        mImpl = std::make_unique<Impl>();
        mImpl->mTerrainSize = params.mTerrainSize;
        mImpl->mTerrainAmplitude = params.mTerrainAmplitude;
        mImpl->mPosition = playerPosition;
        mImpl->mPosition.y =
                TerrainHeight(playerPosition.x, playerPosition.z, params.mTerrainAmplitude);
        return true;
    }

    void Physics::Update(float deltaSeconds, const glm::vec3& horizontalVelocity, bool jump) {
        if (!mImpl) {
            return;
        }
        Impl& state = *mImpl;

        if (state.mOnGround) {
            state.mVelocity.y = jump ? kJumpSpeed : 0.0f;
        }
        state.mVelocity.y += kGravity * deltaSeconds;
        state.mVelocity.x = horizontalVelocity.x;
        state.mVelocity.z = horizontalVelocity.z;

        state.mPosition += state.mVelocity * deltaSeconds;

        const float limit = state.mTerrainSize * 0.5f - kWalkableMargin;
        state.mPosition.x = glm::clamp(state.mPosition.x, -limit, limit);
        state.mPosition.z = glm::clamp(state.mPosition.z, -limit, limit);

        const float ground =
                TerrainHeight(state.mPosition.x, state.mPosition.z, state.mTerrainAmplitude);
        if (state.mPosition.y <= ground) {
            state.mPosition.y = ground;
            state.mVelocity.y = 0.0f;
            state.mOnGround = true;
        } else {
            state.mOnGround = false;
        }
    }

    glm::vec3 Physics::PlayerPosition() const {
        return mImpl ? mImpl->mPosition : glm::vec3(0.0f);
    }

    glm::vec3 Physics::PlayerVelocity() const {
        return mImpl ? mImpl->mVelocity : glm::vec3(0.0f);
    }

    bool Physics::IsOnGround() const {
        return mImpl != nullptr && mImpl->mOnGround;
    }

    void Physics::Destroy() {
        mImpl.reset();
    }
}// namespace hakoniwa
