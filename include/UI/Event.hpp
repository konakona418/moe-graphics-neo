#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace moe::ui {
    // Pointer state for one frame, already mapped into UI space by the caller.
    // The UI never reads raw screen coordinates, so it works identically for a
    // screen-space overlay and a world-space surface (raycast -> UI point).
    struct UiInput {
        glm::vec2 mPointer{-1.0f, -1.0f};
        bool mPrimaryDown{false};
        bool mPrimaryPressed{false};  // edge: became down this frame
        bool mPrimaryReleased{false}; // edge: became up this frame
        float mScroll{0.0f};
        // Set when another UI layer (e.g. the ImGui debug overlay) owns the
        // pointer; the UI then skips hover/press updates but still lays out.
        bool mCaptured{false};
    };

    // Per-frame UI configuration.
    struct UiFrameDesc {
        uint32_t mWidth{0};  // render target size in physical pixels
        uint32_t mHeight{0};
        float mScale{1.0f}; // DPI scale: logical = physical / scale
        UiInput mInput;
        // UI space (logical pixels, y down) -> clip. Unset = the default
        // orthographic mapping of the render target. Set it to apply a layer
        // transform: compose a 3D tilt/perspective here (the caller owns the
        // inverse mapping for input; see the demo's ray-plane unproject).
        std::optional<glm::mat4> mViewProjection;
        // Layer offset: every element is displaced by this UI-space vector,
        // scaled by (1 + the element's Z) so deeper elements move more and
        // z = 0 is the base plane (still moves). Applied before the view
        // projection (so it lives in the UI plane and tilts with the layer) and
        // to hit-testing too, so clicks track the shifted visuals. Direction is
        // the caller's: +x right, +y down; derive it from the pointer/camera.
        glm::vec2 mOffset{0.0f};
    };

    enum class UiEventType { kHoverEnter, kHoverLeave, kPress, kRelease, kClick };

    // One interaction produced by the dispatch pass. Elements with an actionId
    // report it here; C++ callbacks fire alongside. The list is the testable,
    // replayable side of the event model.
    struct UiEvent {
        uint64_t mId{0};
        std::string mAction;
        UiEventType mType{UiEventType::kClick};
    };
}// namespace moe::ui
