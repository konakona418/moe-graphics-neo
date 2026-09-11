#pragma once

#include <UI/Text.hpp>
#include <UI/Types.hpp>

#include <Neo/Assets.hpp>

#include <optional>

namespace moe::ui {
    // Per-element style overrides. Every field is optional; unset fields fall
    // back to the Theme. This is the only style mechanism (no cascade, no
    // selectors).
    struct Style {
        std::optional<glm::vec4> mTextColor;
        std::optional<glm::vec4> mBackground;
        std::optional<glm::vec4> mBorderColor;
        std::optional<float> mRadius;
        std::optional<float> mBorderWidth;
        std::optional<Insets> mPadding;
        std::optional<float> mGap;
        std::optional<float> mFontSize;
        std::optional<neo::Font> mFont;
        std::optional<TextLayout> mTextLayout;
    };

    // Global defaults shared by every element. The app owns one and may swap it
    // per frame.
    struct Theme {
        glm::vec4 mTextColor{0.92f, 0.93f, 0.96f, 1.0f};
        glm::vec4 mBackground{0.13f, 0.14f, 0.18f, 1.0f};
        glm::vec4 mBorderColor{0.30f, 0.33f, 0.42f, 1.0f};
        float mRadius{6.0f};
        float mBorderWidth{0.0f};
        Insets mPadding{12.0f, 12.0f, 12.0f, 12.0f};
        float mGap{8.0f};
        float mFontSize{20.0f};
        float mScrollStep{48.0f}; // pixels scrolled per unit of UiInput::mScroll
        neo::Font mFont;

        glm::vec4 mButtonColor{0.22f, 0.24f, 0.30f, 1.0f};
        glm::vec4 mButtonHover{0.30f, 0.33f, 0.42f, 1.0f};
        glm::vec4 mButtonActive{0.18f, 0.40f, 0.70f, 1.0f};
        glm::vec4 mButtonText{0.95f, 0.96f, 1.0f, 1.0f};
    };

    // A Style resolved against a Theme: every field concrete. Layout and paint
    // read this instead of touching optional fields.
    struct ResolvedStyle {
        glm::vec4 mTextColor;
        glm::vec4 mBackground;
        glm::vec4 mBorderColor;
        float mRadius;
        float mBorderWidth;
        Insets mPadding;
        float mGap;
        float mFontSize;
        neo::Font mFont;
        TextLayout mTextLayout{TextLayout::kSingleLine};
    };

    ResolvedStyle Resolve(const Theme& theme, const Style& style);
}// namespace moe::ui
