#pragma once

#include <Core/SmallVector.hpp>

#include <cstdint>
#include <memory>
#include <string>

// GLFW's opaque window type (forward-declared globally so the callback
// signatures stay GLFW-header-free).
struct GLFWwindow;

namespace moe::neo {
    class Window;

    // Key codes use GLFW key code values (stable across platforms). The enum
    // covers what demos typically use; any other GLFW code works too (the
    // state table spans the full GLFW range).
    enum class KeyCode : int32_t {
        kSpace = 32,
        kA = 65, kB = 66, kC = 67, kD = 68, kE = 69, kF = 70, kG = 71,
        kH = 72, kI = 73, kJ = 74, kK = 75, kL = 76, kM = 77, kN = 78,
        kO = 79, kP = 80, kQ = 81, kR = 82, kS = 83, kT = 84, kU = 85,
        kV = 86, kW = 87, kX = 88, kY = 89, kZ = 90,
        k0 = 48, k1 = 49, k2 = 50, k3 = 51, k4 = 52,
        k5 = 53, k6 = 54, k7 = 55, k8 = 56, k9 = 57,
        kEscape = 256, kEnter = 257, kTab = 258, kBackspace = 259,
        kInsert = 260, kDelete = 261, kRight = 262, kLeft = 263,
        kDown = 264, kUp = 265,
        kLShift = 340, kLCtrl = 341, kLAlt = 342, kRShift = 344,
        kRCtrl = 345, kRAlt = 346,
    };

    struct MouseState {
        float mX{0.0f};     // cursor position (window pixels, y down)
        float mY{0.0f};
        float mDeltaX{0.0f}; // accumulated since the last EndFrame
        float mDeltaY{0.0f};
        float mScrollY{0.0f}; // accumulated scroll since the last EndFrame
        bool mButtonDown[3]{};
        bool mButtonPressed[3]{}; // edge: became down this frame
        bool mButtonReleased[3]{}; // edge: became up this frame
        bool mCaptured{false};     // cursor hidden + locked (FPS look)
    };

    // Event-state + action-binding input system (GLFW-driven). Replaces the old
    // engine's InputBus/proxy design: GLFW callbacks write directly into a
    // fixed key-state table (no event queue, so no lost-key events), and
    // actions map to one or more keys. No proxies or priorities: demos query
    // the state/actions directly; ImGui handles its own capture.
    class Input {
    public:
        Input();
        ~Input();

        Input(const Input&) = delete;
        Input& operator=(const Input&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the input
        // was initialized but not destroyed (leak trap).
        void Destroy();

        // Installs GLFW callbacks on the window (chains with any previous
        // callbacks, e.g. ImGui's). Call once, before DebugUI init.
        bool Init(Window& window, std::string& error);

        // Clears the per-frame edges and accumulated mouse deltas. Call once
        // at the end of each frame (after the demo consumed the state).
        void EndFrame();

        // ---- raw key state (key codes = GLFW values) ----
        bool IsKeyDown(int32_t key) const;
        bool IsKeyJustPressed(int32_t key) const;
        bool IsKeyJustReleased(int32_t key) const;

        // ---- actions ----
        // Binds a key to an action (deduplicated). actionId is an index into a
        // fixed-size table (see kMaxActions).
        bool BindAction(uint32_t actionId, int32_t key);
        void UnbindAction(uint32_t actionId, int32_t key);

        bool IsActionDown(uint32_t actionId) const;
        bool IsActionJustPressed(uint32_t actionId) const;
        bool IsActionJustReleased(uint32_t actionId) const;

        // ---- mouse ----
        const MouseState& GetMouse() const;

        // Locks the cursor to the window (FPS look); false restores it.
        void SetMouseCaptured(bool captured);

        static constexpr uint32_t kMaxActions = 64;

    private:
        // GLFW callbacks (registered in Init; chained with previous ones).
        static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
        static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
