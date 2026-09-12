#pragma once

#include <glm/glm.hpp>

#include <memory>

// Hand-written character physics: gravity, jumping and a ground test against
// the analytic terrain height. Deliberately dependency-free (no Jolt).

namespace hakoniwa {
    struct PhysicsParams {
        float mTerrainSize{600.0f};
        float mTerrainAmplitude{20.0f};
    };

    class Physics {
    public:
        Physics();
        ~Physics();

        Physics(const Physics&) = delete;
        Physics& operator=(const Physics&) = delete;

        bool Init(const PhysicsParams& params, const glm::vec3& playerPosition);
        void Destroy();

        // `horizontalVelocity` is the desired world-space horizontal velocity
        // (y ignored); `jump` requests a jump.
        void Update(float deltaSeconds, const glm::vec3& horizontalVelocity, bool jump);

        glm::vec3 PlayerPosition() const;
        glm::vec3 PlayerVelocity() const;
        bool IsOnGround() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace hakoniwa
