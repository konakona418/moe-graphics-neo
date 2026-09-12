#pragma once

#include <Neo/Input.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

// Free-fly camera: WASD/QE to move, right-drag to look, Shift to sprint. The
// app owns the mouse-capture flag (so ImGui can keep the cursor when a panel is
// hovered) and passes it in each frame.
namespace hakoniwa {
    struct FreeFlyCamera {
        glm::vec3 mPosition{0.0f, 12.0f, 45.0f};
        float mYaw{0.0f};    // radians; 0 looks toward -Z
        float mPitch{-0.05f};
        float mFovYDeg{60.0f};
        float mNear{0.5f};
        float mFar{6000.0f};
        float mSpeed{20.0f};
        float mSensitivity{0.0025f};

        glm::vec3 Forward() const {
            return glm::normalize(glm::vec3(std::cos(mPitch) * std::sin(mYaw),
                    std::sin(mPitch), -std::cos(mPitch) * std::cos(mYaw)));
        }

        glm::vec3 Right() const {
            return glm::normalize(glm::cross(Forward(), glm::vec3(0.0f, 1.0f, 0.0f)));
        }

        glm::vec3 Up() const { return glm::cross(Right(), Forward()); }

        glm::mat4 View() const {
            return glm::lookAt(mPosition, mPosition + Forward(), glm::vec3(0.0f, 1.0f, 0.0f));
        }

        glm::mat4 Projection(float aspect) const {
            glm::mat4 proj = glm::perspective(glm::radians(mFovYDeg), aspect, mNear, mFar);
            proj[1][1] *= -1.0f;
            return proj;
        }

        float TanHalfFov() const { return std::tan(glm::radians(mFovYDeg) * 0.5f); }

        void Look(const moe::neo::Input& input, bool captured) {
            if (captured) {
                const moe::neo::MouseState& mouse = input.GetMouse();
                mYaw -= mouse.mDeltaX * mSensitivity;
                mPitch = glm::clamp(mPitch - mouse.mDeltaY * mSensitivity, -1.5f, 1.5f);
            }
        }

        void MoveFly(const moe::neo::Input& input, float deltaSeconds) {
            const glm::vec3 forward = Forward();
            const glm::vec3 right = Right();
            const float boost = input.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kLShift))
                    ? 4.0f
                    : 1.0f;
            const float step = mSpeed * boost * deltaSeconds;

            if (input.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kW))) {
                mPosition += forward * step;
            }
            if (input.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kS))) {
                mPosition -= forward * step;
            }
            if (input.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kA))) {
                mPosition -= right * step;
            }
            if (input.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kD))) {
                mPosition += right * step;
            }
            if (input.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kE))) {
                mPosition.y += step;
            }
            if (input.IsKeyDown(static_cast<int32_t>(moe::neo::KeyCode::kQ))) {
                mPosition.y -= step;
            }
        }

        void Update(const moe::neo::Input& input, float deltaSeconds, bool captured) {
            Look(input, captured);
            MoveFly(input, deltaSeconds);
        }
    };
}// namespace hakoniwa
