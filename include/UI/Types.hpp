#pragma once

#include <glm/glm.hpp>

namespace moe::ui {
    // Layout size request. Fixed = explicit logical pixels, Fit = intrinsic
    // (measure content), Grow = fill the leftover space along the parent's main
    // axis (distributed by weight among Grow siblings).
    struct Size {
        enum class Kind { kFixed, kFit, kGrow };

        Kind mKind{Kind::kFit};
        float mValue{0.0f}; // Fixed: pixels; Grow: weight

        static Size Fixed(float value) { return {Kind::kFixed, value}; }
        static Size Fit() { return {Kind::kFit, 0.0f}; }
        static Size Grow(float weight = 1.0f) { return {Kind::kGrow, weight}; }
    };

    // Edge insets (padding/margin), in logical pixels.
    struct Insets {
        float mLeft{0.0f};
        float mTop{0.0f};
        float mRight{0.0f};
        float mBottom{0.0f};

        static Insets All(float value) { return {value, value, value, value}; }
        static Insets Symmetric(float horizontal, float vertical) {
            return {horizontal, vertical, horizontal, vertical};
        }
    };

    // Axis-aligned rectangle in UI space (logical pixels, y down).
    struct Rect {
        glm::vec2 mMin{0.0f};
        glm::vec2 mMax{0.0f};

        float Width() const { return mMax.x - mMin.x; }
        float Height() const { return mMax.y - mMin.y; }
        glm::vec2 Size() const { return mMax - mMin; }
        glm::vec2 Center() const { return (mMin + mMax) * 0.5f; }

        bool Contains(const glm::vec2& p) const {
            return p.x >= mMin.x && p.x <= mMax.x && p.y >= mMin.y && p.y <= mMax.y;
        }

        Rect Inset(const Insets& insets) const {
            return {{mMin.x + insets.mLeft, mMin.y + insets.mTop},
                    {mMax.x - insets.mRight, mMax.y - insets.mBottom}};
        }

        Rect Offset(const glm::vec2& delta) const { return {mMin + delta, mMax + delta}; }
    };

    // Cross-axis placement of a child inside its parent.
    enum class Alignment { kStart, kCenter, kEnd, kStretch };

    // Main-axis placement of children inside a Row/Column.
    enum class Justify { kStart, kCenter, kEnd, kSpaceBetween };
}// namespace moe::ui
