#include "UiInternal.hpp"

#include <Neo/Font.hpp>

#include <algorithm>
#include <cstddef>

namespace moe::ui {
    namespace {
        uint32_t DecodeUtf8(std::string_view text, size_t& index) {
            const auto c = static_cast<unsigned char>(text[index]);
            if (c < 0x80) {
                ++index;
                return c;
            }
            uint32_t codepoint = 0;
            int extra = 0;
            if ((c & 0xE0) == 0xC0) {
                codepoint = c & 0x1Fu;
                extra = 1;
            } else if ((c & 0xF0) == 0xE0) {
                codepoint = c & 0x0Fu;
                extra = 2;
            } else if ((c & 0xF8) == 0xF0) {
                codepoint = c & 0x07u;
                extra = 3;
            } else {
                ++index;
                return c;
            }
            ++index;
            for (int i = 0; i < extra && index < text.size(); ++i, ++index) {
                codepoint = (codepoint << 6) | (static_cast<unsigned char>(text[index]) & 0x3Fu);
            }
            return codepoint;
        }

        // Places `size` inside `box` per the axis alignments (Stretch fills).
        Rect AlignRect(const Rect& box, const glm::vec2& size, Alignment horizontal,
                Alignment vertical) {
            const float width = horizontal == Alignment::kStretch ? box.Width() : size.x;
            const float height = vertical == Alignment::kStretch ? box.Height() : size.y;
            float x = box.mMin.x;
            if (horizontal == Alignment::kCenter) {
                x = box.mMin.x + (box.Width() - width) * 0.5f;
            } else if (horizontal == Alignment::kEnd) {
                x = box.mMax.x - width;
            }
            float y = box.mMin.y;
            if (vertical == Alignment::kCenter) {
                y = box.mMin.y + (box.Height() - height) * 0.5f;
            } else if (vertical == Alignment::kEnd) {
                y = box.mMax.y - height;
            }
            return {{x, y}, {x + width, y + height}};
        }
    }// namespace

    glm::vec2 MeasureText(const neo::FontData& font, std::string_view text, float pixelSize) {
        const float scale = font.GetScaleForPixelHeight(pixelSize);
        const float lineHeight = font.GetLineAdvance() * scale;
        float maxWidth = 0.0f;
        float lineWidth = 0.0f;
        uint32_t previous = 0;
        bool hasPrevious = false;
        int lines = 1;
        for (size_t i = 0; i < text.size();) {
            const uint32_t codepoint = DecodeUtf8(text, i);
            if (codepoint == '\n') {
                maxWidth = std::max(maxWidth, lineWidth);
                lineWidth = 0.0f;
                hasPrevious = false;
                ++lines;
                continue;
            }
            const neo::Glyph* glyph = font.FindGlyph(codepoint);
            if (glyph == nullptr) {
                continue;
            }
            if (hasPrevious) {
                lineWidth += font.GetKernAdvance(previous, codepoint) * scale;
            }
            lineWidth += glyph->mAdvance * scale;
            previous = codepoint;
            hasPrevious = true;
        }
        maxWidth = std::max(maxWidth, lineWidth);
        return {maxWidth, lineHeight * static_cast<float>(lines)};
    }

    glm::vec2 Ui::Impl::Measure(const Element& element, const ResolvedStyle& style,
            const glm::vec2& available) {
        glm::vec2 intrinsic{0.0f};
        std::visit(
                [&](const auto& body) {
                    using T = std::decay_t<decltype(body)>;
                    if constexpr (std::is_same_v<T, LabelData>) {
                        if (const neo::FontData* font = style.mFont.GetData()) {
                            intrinsic = MeasureText(*font, body.mText, style.mFontSize);
                        }
                    } else if constexpr (std::is_same_v<T, ImageData>) {
                        if (neo::UploadedTexture* texture = mAssets->GetTexture(body.mTexture)) {
                            intrinsic = {static_cast<float>(texture->mImage.GetWidth()),
                                    static_cast<float>(texture->mImage.GetHeight())};
                        }
                    } else if constexpr (std::is_same_v<T, ButtonData>) {
                        if (const neo::FontData* font = style.mFont.GetData()) {
                            intrinsic = MeasureText(*font, body.mText, style.mFontSize);
                        }
                    } else if constexpr (std::is_same_v<T, SpacerData>) {
                        intrinsic = body.mSize.mKind == Size::Kind::kFixed
                                ? glm::vec2(body.mSize.mValue)
                                : glm::vec2(0.0f);
                    } else if constexpr (std::is_same_v<T, RowData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        intrinsic = MeasureLinear(body.mChildren, style, available, true, gap);
                    } else if constexpr (std::is_same_v<T, ColumnData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        intrinsic = MeasureLinear(body.mChildren, style, available, false, gap);
                    } else if constexpr (std::is_same_v<T, PanelData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        intrinsic = MeasureLinear(body.mChildren, style, available, false, gap);
                    } else if constexpr (std::is_same_v<T, StackData>) {
                        for (const Element& child : body.mChildren) {
                            if (!child.mVisible) {
                                continue;
                            }
                            const ResolvedStyle childStyle = Resolve(mTheme, ElementStyle(child));
                            intrinsic = glm::max(intrinsic,
                                    Measure(child, childStyle, available));
                        }
                    } else if constexpr (std::is_same_v<T, PaddingData>) {
                        if (!body.mChildren.empty()) {
                            const Element& child = body.mChildren.front();
                            const ResolvedStyle childStyle = Resolve(mTheme, ElementStyle(child));
                            const glm::vec2 size = Measure(child, childStyle, available);
                            intrinsic = size + glm::vec2(body.mInsets.mLeft + body.mInsets.mRight,
                                                body.mInsets.mTop + body.mInsets.mBottom);
                        }
                    } else if constexpr (std::is_same_v<T, AlignData>
                            || std::is_same_v<T, ExpandData>) {
                        if (!body.mChildren.empty()) {
                            const Element& child = body.mChildren.front();
                            const ResolvedStyle childStyle = Resolve(mTheme, ElementStyle(child));
                            intrinsic = Measure(child, childStyle, available);
                        }
                    }
                },
                element.mBody);

        intrinsic.x += style.mPadding.mLeft + style.mPadding.mRight;
        intrinsic.y += style.mPadding.mTop + style.mPadding.mBottom;

        const float width = element.mWidth.mKind == Size::Kind::kFixed ? element.mWidth.mValue
                                                                       : intrinsic.x;
        const float height = element.mHeight.mKind == Size::Kind::kFixed ? element.mHeight.mValue
                                                                         : intrinsic.y;
        return {width, height};
    }

    glm::vec2 Ui::Impl::MeasureLinear(const std::vector<Element>& children,
            const ResolvedStyle& style, const glm::vec2& available, bool horizontal, float gap) {
        float main = 0.0f;
        float cross = 0.0f;
        int count = 0;
        for (const Element& child : children) {
            if (!child.mVisible) {
                continue;
            }
            const ResolvedStyle childStyle = Resolve(mTheme, ElementStyle(child));
            const glm::vec2 size = Measure(child, childStyle, available);
            main += horizontal ? size.x : size.y;
            cross = std::max(cross, horizontal ? size.y : size.x);
            ++count;
        }
        main += gap * static_cast<float>(std::max(0, count - 1));
        return horizontal ? glm::vec2(main, cross) : glm::vec2(cross, main);
    }

    void Ui::Impl::Arrange(uint32_t index, const Rect& rect) {
        UiNode& node = mNodes[index];
        node.mRect = rect;
        const Element& element = *node.mElement;
        const ResolvedStyle& style = node.mStyle;
        const Rect content = rect.Inset(style.mPadding);

        std::visit(
                [&](const auto& body) {
                    using T = std::decay_t<decltype(body)>;
                    if constexpr (std::is_same_v<T, RowData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        ArrangeLinear(node, content, body.mChildren, true, body.mJustify,
                                body.mAlign, gap);
                    } else if constexpr (std::is_same_v<T, ColumnData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        ArrangeLinear(node, content, body.mChildren, false, body.mJustify,
                                body.mAlign, gap);
                    } else if constexpr (std::is_same_v<T, PanelData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        ArrangeLinear(node, content, body.mChildren, false, Justify::kStart,
                                Alignment::kStretch, gap);
                    } else if constexpr (std::is_same_v<T, StackData>) {
                        for (uint32_t childIndex : node.mChildren) {
                            const Element& child = *mNodes[childIndex].mElement;
                            const ResolvedStyle childStyle =
                                    Resolve(mTheme, ElementStyle(child));
                            const glm::vec2 size = Measure(child, childStyle, content.Size());
                            const Rect childRect = body.mAlign == Alignment::kStretch
                                    ? content
                                    : AlignRect(content, size, body.mAlign, body.mAlign);
                            Arrange(childIndex, childRect);
                        }
                    } else if constexpr (std::is_same_v<T, PaddingData>) {
                        if (!node.mChildren.empty()) {
                            Arrange(node.mChildren.front(), content.Inset(body.mInsets));
                        }
                    } else if constexpr (std::is_same_v<T, AlignData>) {
                        if (!node.mChildren.empty()) {
                            const uint32_t childIndex = node.mChildren.front();
                            const ResolvedStyle childStyle =
                                    Resolve(mTheme, ElementStyle(*mNodes[childIndex].mElement));
                            const glm::vec2 size =
                                    Measure(*mNodes[childIndex].mElement, childStyle, content.Size());
                            Arrange(childIndex, AlignRect(content, size, body.mAlign, body.mAlign));
                        }
                    } else if constexpr (std::is_same_v<T, ExpandData>) {
                        if (!node.mChildren.empty()) {
                            Arrange(node.mChildren.front(), content);
                        }
                    }
                },
                element.mBody);
    }

    void Ui::Impl::ArrangeLinear(const UiNode& node, const Rect& content,
            const std::vector<Element>& children, bool horizontal, Justify justify,
            Alignment align, float gap) {
        const uint32_t count = static_cast<uint32_t>(node.mChildren.size());
        if (count == 0) {
            return;
        }

        std::vector<glm::vec2> sizes(count);
        float totalMain = 0.0f;
        float totalWeight = 0.0f;
        for (uint32_t i = 0; i < count; ++i) {
            const Element& child = *mNodes[node.mChildren[i]].mElement;
            sizes[i] = Measure(child, mNodes[node.mChildren[i]].mStyle, content.Size());
            const Size& mainSize = horizontal ? child.mWidth : child.mHeight;
            if (mainSize.mKind == Size::Kind::kGrow) {
                totalWeight += std::max(mainSize.mValue, 0.0f);
            } else {
                totalMain += horizontal ? sizes[i].x : sizes[i].y;
            }
        }
        totalMain += gap * static_cast<float>(count - 1);
        const float contentMain = horizontal ? content.Width() : content.Height();
        const float leftover = std::max(0.0f, contentMain - totalMain);

        float offset = 0.0f;
        float extraGap = 0.0f;
        switch (justify) {
            case Justify::kStart: break;
            case Justify::kCenter: offset = leftover * 0.5f; break;
            case Justify::kEnd: offset = leftover; break;
            case Justify::kSpaceBetween:
                if (count > 1) {
                    extraGap = leftover / static_cast<float>(count - 1);
                }
                break;
        }

        const float contentCross = horizontal ? content.Height() : content.Width();
        float cursor = (horizontal ? content.mMin.x : content.mMin.y) + offset;
        for (uint32_t i = 0; i < count; ++i) {
            const Element& child = *mNodes[node.mChildren[i]].mElement;
            const Size& mainSize = horizontal ? child.mWidth : child.mHeight;
            float main = horizontal ? sizes[i].x : sizes[i].y;
            if (mainSize.mKind == Size::Kind::kGrow && totalWeight > 0.0f) {
                main = leftover * (mainSize.mValue / totalWeight);
            }
            float cross = horizontal ? sizes[i].y : sizes[i].x;
            if (align == Alignment::kStretch) {
                cross = contentCross;
            }
            float crossPos = horizontal ? content.mMin.y : content.mMin.x;
            if (align == Alignment::kCenter) {
                crossPos += (contentCross - cross) * 0.5f;
            } else if (align == Alignment::kEnd) {
                crossPos += contentCross - cross;
            }
            const Rect childRect = horizontal
                    ? Rect{{cursor, crossPos}, {cursor + main, crossPos + cross}}
                    : Rect{{crossPos, cursor}, {crossPos + cross, cursor + main}};
            Arrange(node.mChildren[i], childRect);
            cursor += main + gap + extraGap;
        }
    }
}// namespace moe::ui
