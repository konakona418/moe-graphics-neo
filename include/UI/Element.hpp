#pragma once

#include <UI/Style.hpp>
#include <UI/Types.hpp>

#include <Neo/Assets.hpp>

#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace moe::ui {
    struct Element;

    // ---- leaf payloads ----

    struct LabelData {
        std::string mText;
        Alignment mAlign{Alignment::kStart};
        Style mStyle;
    };

    struct ImageData {
        neo::TextureHandle mTexture;
        glm::vec4 mTint{1.0f};
        Style mStyle;
    };

    struct ButtonData {
        std::string mText;
        std::function<void()> mOnClick;
        std::string mAction;
        Style mStyle;
    };

    struct SpacerData {
        Size mSize{Size::Grow(1.0f)};
    };

    // ---- container payloads ----

    struct RowData {
        std::vector<Element> mChildren;
        float mGap{-1.0f}; // < 0 = theme gap
        Justify mJustify{Justify::kStart};
        Alignment mAlign{Alignment::kStretch};
        Style mStyle;
    };

    struct ColumnData {
        std::vector<Element> mChildren;
        float mGap{-1.0f};
        Justify mJustify{Justify::kStart};
        Alignment mAlign{Alignment::kStretch};
        Style mStyle;
    };

    struct StackData {
        std::vector<Element> mChildren;
        Alignment mAlign{Alignment::kStretch};
        Style mStyle;
    };

    // Background + border + padding around a vertical (column) run of children.
    struct PanelData {
        std::vector<Element> mChildren;
        float mGap{-1.0f};
        Style mStyle;
    };

    struct PaddingData {
        std::vector<Element> mChildren; // first child used
        Insets mInsets;
    };

    struct AlignData {
        std::vector<Element> mChildren; // first child used
        Alignment mAlign{Alignment::kCenter};
    };

    struct ExpandData {
        std::vector<Element> mChildren; // first child used
        float mWeight{1.0f};
    };

    // A node of the declarative view tree. Plain value type: the tree is
    // rebuilt every frame, never mutated in place across frames. Common fields
    // (size/key/z) live here; the variant carries the element-specific payload.
    struct Element {
        using Body = std::variant<LabelData, ImageData, ButtonData, SpacerData, RowData,
                ColumnData, StackData, PanelData, PaddingData, AlignData, ExpandData>;

        Body mBody;
        Size mWidth{Size::Fit()};
        Size mHeight{Size::Fit()};
        std::string mKey; // stable id override (lists, reorder)
        float mZ{0.0f};   // layer-offset depth (0 = base plane, positive moves more)
        bool mVisible{true};
        bool mEnabled{true};

        Element() = default;

        template<typename T>
            requires(!std::is_same_v<std::decay_t<T>, Element>)
        Element(T&& body)
            : mBody(std::forward<T>(body)) {}

        Element& SetKey(std::string key) {
            mKey = std::move(key);
            return *this;
        }

        Element& SetWidth(Size width) {
            mWidth = width;
            return *this;
        }

        Element& SetHeight(Size height) {
            mHeight = height;
            return *this;
        }

        Element& SetSize(Size width, Size height) {
            mWidth = width;
            mHeight = height;
            return *this;
        }

        Element& SetZ(float z) {
            mZ = z;
            return *this;
        }

        Element& SetEnabled(bool enabled) {
            mEnabled = enabled;
            return *this;
        }
    };

    // ---- builders (the daily API) ----

    namespace detail {
        // Containers and leaves default to no padding; Panel/Button keep the
        // theme padding.
        inline Style& NoPadding(Style& style) {
            if (!style.mPadding) {
                style.mPadding = Insets::All(0.0f);
            }
            return style;
        }
    }// namespace detail

    inline Element Label(std::string text, Style style = {}) {
        LabelData data;
        data.mText = std::move(text);
        data.mStyle = detail::NoPadding(style);
        return data;
    }

    inline Element Image(neo::TextureHandle texture, Style style = {}) {
        ImageData data;
        data.mTexture = texture;
        data.mStyle = detail::NoPadding(style);
        return data;
    }

    inline Element Button(std::string text, std::function<void()> onClick = {},
            std::string action = {}, Style style = {}) {
        ButtonData data;
        data.mText = std::move(text);
        data.mOnClick = std::move(onClick);
        data.mAction = std::move(action);
        data.mStyle = std::move(style);
        return data;
    }

    inline Element Spacer(Size size = Size::Grow(1.0f)) {
        SpacerData data;
        data.mSize = size;
        return data;
    }

    inline Element Row(std::vector<Element> children, Style style = {}) {
        RowData data;
        data.mChildren = std::move(children);
        data.mStyle = detail::NoPadding(style);
        return data;
    }

    inline Element Column(std::vector<Element> children, Style style = {}) {
        ColumnData data;
        data.mChildren = std::move(children);
        data.mStyle = detail::NoPadding(style);
        return data;
    }

    inline Element Stack(std::vector<Element> children, Style style = {}) {
        StackData data;
        data.mChildren = std::move(children);
        data.mStyle = detail::NoPadding(style);
        return data;
    }

    inline Element Panel(std::vector<Element> children, Style style = {}) {
        PanelData data;
        data.mChildren = std::move(children);
        data.mStyle = std::move(style);
        return data;
    }

    inline Element Padding(Element child, Insets insets) {
        PaddingData data;
        data.mChildren.push_back(std::move(child));
        data.mInsets = insets;
        return data;
    }

    inline Element Align(Element child, Alignment align = Alignment::kCenter) {
        AlignData data;
        data.mChildren.push_back(std::move(child));
        data.mAlign = align;
        return data;
    }

    inline Element Expand(Element child, float weight = 1.0f) {
        ExpandData data;
        data.mChildren.push_back(std::move(child));
        data.mWeight = weight;
        return data;
    }
}// namespace moe::ui
