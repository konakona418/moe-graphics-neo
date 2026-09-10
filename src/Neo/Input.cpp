#include "Neo/Input.hpp"

#include <Core/Error.hpp>
#include "Neo/Window.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>

namespace moe::neo {
    namespace {
        // GLFW key code range: 0 .. GLFW_KEY_LAST. Sized generously; the extra
        // slots stay zeroed.
        constexpr uint32_t kKeyCount = 512;

        // GLFW callbacks receive only the window, so the single App-owned Input
        // is routed through this pointer (there is exactly one active input).
        Input* gActiveInput = nullptr;
    }// namespace

    struct Input::Impl {
        // key state tables indexed by key code
        uint8_t mKeyDown[kKeyCount]{};
        uint8_t mKeyPressed[kKeyCount]{};
        uint8_t mKeyReleased[kKeyCount]{};

        // action -> bound keys (index = actionId)
        struct ActionEntry {
            SmallVector<int32_t, 2> mKeys;
        };
        ActionEntry mActions[kMaxActions];

        MouseState mMouse;
        GLFWwindow* mWindow{nullptr};
        // previous callbacks (chained, e.g. ImGui's)
        GLFWkeyfun mPrevKey{nullptr};
        GLFWmousebuttonfun mPrevMouseButton{nullptr};
        GLFWcursorposfun mPrevCursorPos{nullptr};
        GLFWscrollfun mPrevScroll{nullptr};
        bool mActive{false};
    };

    void Input::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        Input* self = gActiveInput;
        if (self != nullptr && key >= 0 && key < static_cast<int>(kKeyCount)) {
            auto& impl = *self->mImpl;
            if (action == GLFW_PRESS) {
                impl.mKeyDown[key] = 1;
                impl.mKeyPressed[key] = 1;
            } else if (action == GLFW_RELEASE) {
                impl.mKeyDown[key] = 0;
                impl.mKeyReleased[key] = 1;
            }
        }
        if (self != nullptr && self->mImpl->mPrevKey != nullptr) {
            self->mImpl->mPrevKey(window, key, scancode, action, mods);
        }
    }

    void Input::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        Input* self = gActiveInput;
        if (self != nullptr && button >= 0 && button < 3) {
            auto& impl = *self->mImpl;
            if (action == GLFW_PRESS) {
                impl.mMouse.mButtonDown[button] = true;
                impl.mMouse.mButtonPressed[button] = true;
            } else if (action == GLFW_RELEASE) {
                impl.mMouse.mButtonDown[button] = false;
                impl.mMouse.mButtonReleased[button] = true;
            }
        }
        if (self != nullptr && self->mImpl->mPrevMouseButton != nullptr) {
            self->mImpl->mPrevMouseButton(window, button, action, mods);
        }
    }

    void Input::CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
        Input* self = gActiveInput;
        if (self != nullptr) {
            auto& mouse = self->mImpl->mMouse;
            const float x = static_cast<float>(xpos);
            const float y = static_cast<float>(ypos);
            mouse.mDeltaX += x - mouse.mX;
            mouse.mDeltaY += y - mouse.mY;
            mouse.mX = x;
            mouse.mY = y;
        }
        if (self != nullptr && self->mImpl->mPrevCursorPos != nullptr) {
            self->mImpl->mPrevCursorPos(window, xpos, ypos);
        }
    }

    void Input::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        Input* self = gActiveInput;
        if (self != nullptr) {
            self->mImpl->mMouse.mScrollY += static_cast<float>(yoffset);
        }
        if (self != nullptr && self->mImpl->mPrevScroll != nullptr) {
            self->mImpl->mPrevScroll(window, xoffset, yoffset);
        }
    }

    Input::Input() = default;

    Input::~Input() {
        if (mImpl != nullptr) {
            std::fprintf(stderr, "[neo] Input leaked: Destroy() not called\n");
            std::abort();
        }
    }

    void Input::Destroy() {
        if (mImpl == nullptr) {
            return;
        }
        if (mImpl->mActive && mImpl->mWindow != nullptr) {
            glfwSetKeyCallback(mImpl->mWindow, mImpl->mPrevKey);
            glfwSetMouseButtonCallback(mImpl->mWindow, mImpl->mPrevMouseButton);
            glfwSetCursorPosCallback(mImpl->mWindow, mImpl->mPrevCursorPos);
            glfwSetScrollCallback(mImpl->mWindow, mImpl->mPrevScroll);
            if (gActiveInput == this) {
                gActiveInput = nullptr;
            }
        }
        mImpl.reset();
    }

    bool Input::Init(Window& window) {
        mImpl = std::make_unique<Impl>();
        mImpl->mWindow = reinterpret_cast<GLFWwindow*>(window.GetHandle());
        if (mImpl->mWindow == nullptr) {
            mImpl.reset();
            return moe::Fail("Input: no GLFW window");
        }

        // Chain with whatever callbacks are installed (e.g. ImGui's later).
        mImpl->mPrevKey = glfwSetKeyCallback(mImpl->mWindow, &Input::KeyCallback);
        mImpl->mPrevMouseButton = glfwSetMouseButtonCallback(mImpl->mWindow, &Input::MouseButtonCallback);
        mImpl->mPrevCursorPos = glfwSetCursorPosCallback(mImpl->mWindow, &Input::CursorPosCallback);
        mImpl->mPrevScroll = glfwSetScrollCallback(mImpl->mWindow, &Input::ScrollCallback);

        gActiveInput = this;
        mImpl->mActive = true;
        return true;
    }

    void Input::EndFrame() {
        if (mImpl == nullptr) {
            return;
        }
        std::memset(mImpl->mKeyPressed, 0, sizeof(mImpl->mKeyPressed));
        std::memset(mImpl->mKeyReleased, 0, sizeof(mImpl->mKeyReleased));
        std::memset(mImpl->mMouse.mButtonPressed, 0, sizeof(mImpl->mMouse.mButtonPressed));
        std::memset(mImpl->mMouse.mButtonReleased, 0, sizeof(mImpl->mMouse.mButtonReleased));
        mImpl->mMouse.mDeltaX = 0.0f;
        mImpl->mMouse.mDeltaY = 0.0f;
        mImpl->mMouse.mScrollY = 0.0f;
    }

    bool Input::IsKeyDown(int32_t key) const {
        return mImpl != nullptr && key >= 0 && key < static_cast<int32_t>(kKeyCount)
                && mImpl->mKeyDown[key] != 0;
    }

    bool Input::IsKeyJustPressed(int32_t key) const {
        return mImpl != nullptr && key >= 0 && key < static_cast<int32_t>(kKeyCount)
                && mImpl->mKeyPressed[key] != 0;
    }

    bool Input::IsKeyJustReleased(int32_t key) const {
        return mImpl != nullptr && key >= 0 && key < static_cast<int32_t>(kKeyCount)
                && mImpl->mKeyReleased[key] != 0;
    }

    bool Input::BindAction(uint32_t actionId, int32_t key) {
        if (mImpl == nullptr || actionId >= kMaxActions || key < 0
                || key >= static_cast<int32_t>(kKeyCount)) {
            return false;
        }
        auto& entry = mImpl->mActions[actionId];
        for (const int32_t existing : entry.mKeys) {
            if (existing == key) {
                return true; // already bound
            }
        }
        entry.mKeys.push_back(key);
        return true;
    }

    void Input::UnbindAction(uint32_t actionId, int32_t key) {
        if (mImpl == nullptr || actionId >= kMaxActions) {
            return;
        }
        auto& entry = mImpl->mActions[actionId];
        for (size_t i = 0; i < entry.mKeys.size(); ++i) {
            if (entry.mKeys[i] == key) {
                // order does not matter for bindings: swap with the back
                entry.mKeys[i] = entry.mKeys.back();
                entry.mKeys.pop_back();
                return;
            }
        }
    }

    bool Input::IsActionDown(uint32_t actionId) const {
        if (mImpl == nullptr || actionId >= kMaxActions) {
            return false;
        }
        for (const int32_t key : mImpl->mActions[actionId].mKeys) {
            if (IsKeyDown(key)) {
                return true;
            }
        }
        return false;
    }

    bool Input::IsActionJustPressed(uint32_t actionId) const {
        if (mImpl == nullptr || actionId >= kMaxActions) {
            return false;
        }
        for (const int32_t key : mImpl->mActions[actionId].mKeys) {
            if (IsKeyJustPressed(key)) {
                return true;
            }
        }
        return false;
    }

    bool Input::IsActionJustReleased(uint32_t actionId) const {
        if (mImpl == nullptr || actionId >= kMaxActions) {
            return false;
        }
        for (const int32_t key : mImpl->mActions[actionId].mKeys) {
            if (IsKeyJustReleased(key)) {
                return true;
            }
        }
        return false;
    }

    const MouseState& Input::GetMouse() const {
        static const MouseState kEmpty{};
        return mImpl != nullptr ? mImpl->mMouse : kEmpty;
    }

    void Input::SetMouseCaptured(bool captured) {
        if (mImpl == nullptr || mImpl->mWindow == nullptr) {
            return;
        }
        mImpl->mMouse.mCaptured = captured;
        glfwSetInputMode(mImpl->mWindow, GLFW_CURSOR,
                captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
}// namespace moe::neo
