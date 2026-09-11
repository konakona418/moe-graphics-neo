#pragma once

#include <UI/Ui.hpp>

#include <Neo/Assets.hpp>
#include <Neo/Renderer.hpp>

#include <RHI/Buffer.hpp>

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace moe::ui {
    // One flattened node of the frame's view tree. The element pointer stays
    // valid for the duration of EndFrame (the caller's tree is alive then).
    struct UiNode {
        const Element* mElement{nullptr};
        ResolvedStyle mStyle;
        uint64_t mId{0};
        int32_t mParent{-1};
        std::vector<uint32_t> mChildren;
        Rect mRect; // border box in logical pixels, filled by arrange
        // Effective clip (offset space, logical pixels): the intersection of
        // every clipping ancestor's box. Empty/false when unclipped. Used by
        // hit-testing (a clipped-away point must not hit) and by paint.
        Rect mClip;
        bool mHasClip{false};
        // Corner radius of the clip shape. The scissor is a rectangle, so it is
        // inset by this much on every side (the largest axis-aligned rectangle
        // inside the rounded clip): no corner leak, at the cost of trimming the
        // straight edges by the same amount.
        float mClipRadius{0.0f};
        bool mHovered{false};
    };

    // Vertex layout matching shaders/slang/examples/ui.slang.
    struct UiVertex {
        glm::vec2 mPos;
        glm::vec2 mUv;
        glm::vec4 mColor;
        glm::vec4 mRect;   // center.xy, halfSize.xy
        glm::vec4 mParams; // mode, radius, borderWidth, z
    };

    struct UiRectBatch {
        neo::TextureHandle mTexture;
        uint32_t mFirstVertex{0};
        uint32_t mVertexCount{0};
        Rect mClip;
        bool mHasClip{false};
        float mClipRadius{0.0f};
    };

    struct UiTextCmd {
        neo::Font mFont;
        std::string mText;
        glm::vec4 mColor;
        float mFontSize{0.0f};
        glm::vec2 mPos{0.0f};
        float mZ{0.0f};
        Rect mClip;
        bool mHasClip{false};
        float mClipRadius{0.0f};
    };

    // Paint order is a single command list: rect batches and text draws
    // interleave exactly as the tree was walked, so a later element's
    // background occludes an earlier element's text (and vice versa).
    enum class UiCmdType { kRects, kText };

    struct UiCmd {
        UiCmdType mType{UiCmdType::kRects};
        uint32_t mIndex{0}; // into mBatches or mTexts
    };

    // Calls `fn(child)` for every child element of `element` (no-op for leaves).
    template<typename F>
    void ForEachChild(const Element& element, F&& fn) {
        std::visit(
                [&](const auto& body) {
                    using T = std::decay_t<decltype(body)>;
                    if constexpr (requires { body.mChildren; }) {
                        for (const Element& child : body.mChildren) {
                            fn(child);
                        }
                    }
                },
                element.mBody);
    }

    // Extracts the payload's Style (empty for payloads without one).
    Style ElementStyle(const Element& element);

    // True when the element reacts to the pointer (has a button payload).
    bool IsInteractive(const Element& element);

    struct Ui::Impl {
        neo::Assets* mAssets{nullptr};
        neo::Renderer* mRenderer{nullptr};
        rhi::Device* mDevice{nullptr};
        Theme mTheme;
        UiFrameDesc mFrame;
        std::vector<UiEvent> mEvents;

        neo::ProgramHandle mProgram;     // ui.slang (rounded rects/textures)
        neo::ProgramHandle mTextProgram; // text.slang (glyph outlines)
        neo::TextureHandle mWhite;       // 1x1 white, sampled by solid shapes
        neo::RenderTargetHandle mTarget;
        uint32_t mTargetWidth{0};
        uint32_t mTargetHeight{0};

        rhi::Buffer mVertexBuffer;
        void* mVertexMapped{nullptr};
        uint32_t mVertexCapacity{0};

        std::vector<UiNode> mNodes;
        std::unordered_map<uint64_t, bool> mHoverState; // persisted across frames
        std::unordered_map<uint64_t, float> mScrollState; // persisted across frames
        uint64_t mActiveId{0};

        std::vector<UiVertex> mVertices;
        std::vector<UiRectBatch> mBatches;
        std::vector<UiTextCmd> mTexts;
        std::vector<UiCmd> mCommands;

        bool EnsureResources();
        bool EnsureVertexCapacity(uint32_t vertexCount);
        bool EnsureTarget(uint32_t width, uint32_t height);
        void ResetFrame();

        void Flatten(const Element& element, int32_t parent, uint64_t parentId,
                uint32_t childIndex);
        void Dispatch();

        // Layout.cpp
        glm::vec2 Measure(const Element& element, const ResolvedStyle& style,
                const glm::vec2& available);
        glm::vec2 MeasureLinear(const std::vector<Element>& children, const ResolvedStyle& style,
                const glm::vec2& available, bool horizontal, float gap);
        void Arrange(uint32_t index, const Rect& rect, const Rect* clip = nullptr,
                float clipRadius = 0.0f);
        void ArrangeLinear(const UiNode& node, const Rect& content,
                const std::vector<Element>& children, bool horizontal, Justify justify,
                Alignment align, float gap, const Rect* clip, float clipRadius);

        // Render.cpp
        void BuildDrawList();
        void PushRect(const Rect& rect, const glm::vec4& color, float radius, float borderWidth,
                int mode, float z, neo::TextureHandle texture, const Rect* clip,
                float clipRadius);
        void PushText(const Rect& box, const ResolvedStyle& style, std::string_view text,
                Alignment align, float z, const Rect* clip, float clipRadius);
        void Record(neo::Renderer& renderer);
    };
}// namespace moe::ui
