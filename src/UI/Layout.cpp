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

        // Intersects two rectangles; may be empty (min > max), which callers
        // treat as "clips everything".
        Rect Intersect(const Rect& a, const Rect& b) {
            return {{std::max(a.mMin.x, b.mMin.x), std::max(a.mMin.y, b.mMin.y)},
                    {std::min(a.mMax.x, b.mMax.x), std::min(a.mMax.y, b.mMax.y)}};
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

    // Advance width of one line (no newline handling).
    float LineAdvanceWidth(const neo::FontData& font, std::string_view text, float scale) {
        float width = 0.0f;
        uint32_t previous = 0;
        bool hasPrevious = false;
        for (size_t i = 0; i < text.size();) {
            const uint32_t codepoint = DecodeUtf8(text, i);
            if (codepoint == '\n' || codepoint == '\r') {
                continue;
            }
            const neo::Glyph* glyph = font.FindGlyph(codepoint);
            if (glyph == nullptr) {
                continue;
            }
            if (hasPrevious) {
                width += font.GetKernAdvance(previous, codepoint) * scale;
            }
            width += glyph->mAdvance * scale;
            previous = codepoint;
            hasPrevious = true;
        }
        return width;
    }

    // Measures a text element for the layout pass, applying its TextLayout
    // within the available width.
    glm::vec2 MeasureTextElement(const Element& element, const ResolvedStyle& style,
            std::string_view text, const glm::vec2& available) {
        const float horizontalPadding = style.mPadding.mLeft + style.mPadding.mRight;
        float maxWidth = 0.0f;
        if (style.mTextLayout != TextLayout::kSingleLine) {
            const float width = element.mWidth.mKind == Size::Kind::kFixed
                    ? element.mWidth.mValue
                    : available.x;
            maxWidth = std::max(0.0f, width - horizontalPadding);
        }
        const TextMetrics metrics =
                MeasureText(style.mFont, text, style.mFontSize, style.mTextLayout, maxWidth);
        glm::vec2 size = metrics.mSize;
        if (style.mTextLayout == TextLayout::kWrap && maxWidth > 0.0f
                && element.mWidth.mKind != Size::Kind::kFixed) {
            // Fill the available width so the laid-out box matches the wrap width.
            size.x = maxWidth;
        }
        return size;
    }
    }// namespace

    TextMetrics MeasureText(const neo::Font& fontAsset, std::string_view text, float pixelSize,
            TextLayout layout, float maxWidth) {
        TextMetrics metrics;
        const neo::FontData* fontPtr = fontAsset.GetData();
        if (fontPtr == nullptr) {
            return metrics;
        }
        const neo::FontData& font = *fontPtr;
        const float scale = font.GetScaleForPixelHeight(pixelSize);
        metrics.mLineHeight = font.GetLineAdvance() * scale;
        if (scale <= 0.0f) {
            return metrics;
        }

        // Explicit newlines split the text into paragraphs first.
        std::vector<std::string_view> paragraphs;
        size_t paragraphStart = 0;
        for (size_t i = 0; i <= text.size(); ++i) {
            if (i == text.size() || text[i] == '\n') {
                paragraphs.push_back(text.substr(paragraphStart, i - paragraphStart));
                paragraphStart = i + 1;
            }
        }

        const auto pushLine = [&](std::string line) {
            const float width = LineAdvanceWidth(font, line, scale);
            metrics.mSize.x = std::max(metrics.mSize.x, width);
            metrics.mLines.push_back({std::move(line), width});
        };

        if (layout == TextLayout::kEllipsis && maxWidth > 0.0f) {
            const std::string_view paragraph =
                    paragraphs.empty() ? std::string_view{} : paragraphs.front();
            const float full = LineAdvanceWidth(font, paragraph, scale);
            if (full <= maxWidth) {
                pushLine(std::string(paragraph));
            } else {
                const float dots = LineAdvanceWidth(font, "...", scale);
                std::string line;
                float width = 0.0f;
                uint32_t previous = 0;
                bool hasPrevious = false;
                size_t i = 0;
                while (i < paragraph.size()) {
                    const size_t before = i;
                    const uint32_t codepoint = DecodeUtf8(paragraph, i);
                    const neo::Glyph* glyph = font.FindGlyph(codepoint);
                    if (glyph == nullptr) {
                        continue;
                    }
                    float advance = glyph->mAdvance * scale;
                    if (hasPrevious) {
                        advance += font.GetKernAdvance(previous, codepoint) * scale;
                    }
                    if (width + advance + dots > maxWidth) {
                        break;
                    }
                    line.append(paragraph.substr(before, i - before));
                    width += advance;
                    previous = codepoint;
                    hasPrevious = true;
                }
                line += "...";
                pushLine(std::move(line));
            }
            metrics.mSize.y = metrics.mLineHeight * static_cast<float>(metrics.mLines.size());
            return metrics;
        }

        if (layout == TextLayout::kWrap && maxWidth > 0.0f) {
            for (const std::string_view paragraph : paragraphs) {
                std::vector<std::string_view> words;
                size_t wordStart = 0;
                for (size_t i = 0; i <= paragraph.size(); ++i) {
                    if (i == paragraph.size() || paragraph[i] == ' ') {
                        if (i > wordStart) {
                            words.push_back(paragraph.substr(wordStart, i - wordStart));
                        }
                        wordStart = i + 1;
                    }
                }
                std::string line;
                float lineWidth = 0.0f;
                for (const std::string_view word : words) {
                    const float wordWidth = LineAdvanceWidth(font, word, scale);
                    const float spaceWidth =
                            line.empty() ? 0.0f : LineAdvanceWidth(font, " ", scale);
                    if (!line.empty() && lineWidth + spaceWidth + wordWidth > maxWidth) {
                        pushLine(std::move(line));
                        line.clear();
                        lineWidth = 0.0f;
                    }
                    if (line.empty() && wordWidth > maxWidth) {
                        // Break an oversized word by character.
                        size_t i = 0;
                        while (i < word.size()) {
                            const size_t before = i;
                            const uint32_t codepoint = DecodeUtf8(word, i);
                            const neo::Glyph* glyph = font.FindGlyph(codepoint);
                            if (glyph == nullptr) {
                                continue;
                            }
                            const float advance = glyph->mAdvance * scale;
                            if (!line.empty() && lineWidth + advance > maxWidth) {
                                pushLine(std::move(line));
                                line.clear();
                                lineWidth = 0.0f;
                            }
                            line.append(word.substr(before, i - before));
                            lineWidth += advance;
                        }
                    } else {
                        if (!line.empty()) {
                            line += ' ';
                            lineWidth += spaceWidth;
                        }
                        line += std::string(word);
                        lineWidth += wordWidth;
                    }
                }
                pushLine(std::move(line));
            }
            metrics.mSize.y = metrics.mLineHeight * static_cast<float>(metrics.mLines.size());
            return metrics;
        }

        // kSingleLine (or no width): explicit breaks only.
        for (const std::string_view paragraph : paragraphs) {
            pushLine(std::string(paragraph));
        }
        metrics.mSize.y = metrics.mLineHeight * static_cast<float>(metrics.mLines.size());
        return metrics;
    }

    glm::vec2 Ui::Impl::Measure(const Element& element, const ResolvedStyle& style,
            const glm::vec2& available) {
        glm::vec2 intrinsic{0.0f};
        std::visit(
                [&](const auto& body) {
                    using T = std::decay_t<decltype(body)>;
                    if constexpr (std::is_same_v<T, LabelData>) {
                        intrinsic = MeasureTextElement(element, style, body.mText, available);
                    } else if constexpr (std::is_same_v<T, ImageData>) {
                        if (body.mRawImage != nullptr) {
                            intrinsic = {static_cast<float>(body.mRawImage->GetWidth()),
                                    static_cast<float>(body.mRawImage->GetHeight())};
                        } else if (neo::UploadedTexture* texture =
                                           mAssets->GetTexture(body.mTexture)) {
                            intrinsic = {static_cast<float>(texture->mImage.GetWidth()),
                                    static_cast<float>(texture->mImage.GetHeight())};
                        }
                    } else if constexpr (std::is_same_v<T, ButtonData>) {
                        intrinsic = MeasureTextElement(element, style, body.mText, available);
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
                            || std::is_same_v<T, ExpandData>
                            || std::is_same_v<T, ClipData>
                            || std::is_same_v<T, ScrollViewData>) {
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

    void Ui::Impl::Arrange(uint32_t index, const Rect& rect, const Rect* clip) {
        UiNode& node = mNodes[index];
        node.mRect = rect;
        if (clip != nullptr) {
            node.mClip = *clip;
            node.mHasClip = true;
        }
        const Element& element = *node.mElement;
        const ResolvedStyle& style = node.mStyle;
        const Rect content = rect.Inset(style.mPadding);

        // Clipping containers define a new clip in offset space: the layer
        // offset is applied per element by z, so the box moves with it. The
        // container's own drawing (background/border) is clipped by its
        // ancestors only; children are clipped to the content box (inside the
        // padding/border) by the stencil mask.
        const glm::vec2 offset = mFrame.mOffset * (1.0f + element.mZ);
        const Rect offsetContent = content.Offset(offset);
        const Rect childClip =
                clip != nullptr ? Intersect(*clip, offsetContent) : offsetContent;

        std::visit(
                [&](const auto& body) {
                    using T = std::decay_t<decltype(body)>;
                    if constexpr (std::is_same_v<T, RowData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        ArrangeLinear(node, content, body.mChildren, true, body.mJustify,
                                body.mAlign, gap, clip);
                    } else if constexpr (std::is_same_v<T, ColumnData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        ArrangeLinear(node, content, body.mChildren, false, body.mJustify,
                                body.mAlign, gap, clip);
                    } else if constexpr (std::is_same_v<T, PanelData>) {
                        const float gap = body.mGap >= 0.0f ? body.mGap : style.mGap;
                        ArrangeLinear(node, content, body.mChildren, false, Justify::kStart,
                                Alignment::kStretch, gap, clip);
                    } else if constexpr (std::is_same_v<T, StackData>) {
                        for (uint32_t childIndex : node.mChildren) {
                            const Element& child = *mNodes[childIndex].mElement;
                            const ResolvedStyle childStyle =
                                    Resolve(mTheme, ElementStyle(child));
                            const glm::vec2 size = Measure(child, childStyle, content.Size());
                            const Rect childRect = body.mAlign == Alignment::kStretch
                                    ? content
                                    : AlignRect(content, size, body.mAlign, body.mAlign);
                            Arrange(childIndex, childRect, clip);
                        }
                    } else if constexpr (std::is_same_v<T, PaddingData>) {
                        if (!node.mChildren.empty()) {
                            Arrange(node.mChildren.front(), content.Inset(body.mInsets), clip);
                        }
                    } else if constexpr (std::is_same_v<T, AlignData>) {
                        if (!node.mChildren.empty()) {
                            const uint32_t childIndex = node.mChildren.front();
                            const ResolvedStyle childStyle =
                                    Resolve(mTheme, ElementStyle(*mNodes[childIndex].mElement));
                            const glm::vec2 size =
                                    Measure(*mNodes[childIndex].mElement, childStyle, content.Size());
                            Arrange(childIndex, AlignRect(content, size, body.mAlign, body.mAlign),
                                    clip);
                        }
                    } else if constexpr (std::is_same_v<T, ExpandData>) {
                        if (!node.mChildren.empty()) {
                            Arrange(node.mChildren.front(), content, clip);
                        }
                    } else if constexpr (std::is_same_v<T, ClipData>) {
                        if (!node.mChildren.empty()) {
                            const uint32_t childIndex = node.mChildren.front();
                            const Element& child = *mNodes[childIndex].mElement;
                            const ResolvedStyle childStyle =
                                    Resolve(mTheme, ElementStyle(child));
                            // Keep the child's own size at the content origin so
                            // it can overflow (that is the point of a clip).
                            const glm::vec2 size = Measure(child, childStyle, content.Size());
                            const Rect childRect{{content.mMin.x, content.mMin.y},
                                    {content.mMin.x + size.x, content.mMin.y + size.y}};
                            Arrange(childIndex, childRect, &childClip);
                        }
                    } else if constexpr (std::is_same_v<T, ScrollViewData>) {
                        if (!node.mChildren.empty()) {
                            const uint32_t childIndex = node.mChildren.front();
                            const Element& child = *mNodes[childIndex].mElement;
                            const ResolvedStyle childStyle =
                                    Resolve(mTheme, ElementStyle(child));
                            const glm::vec2 childSize = Measure(child, childStyle, content.Size());
                            float& scroll = mScrollState[node.mId];
                            const float maxScroll =
                                    std::max(0.0f, childSize.y - content.Height());
                            scroll = std::clamp(scroll, 0.0f, maxScroll);
                            const Rect childRect{
                                    {content.mMin.x, content.mMin.y - scroll},
                                    {content.mMin.x + content.Width(),
                                            content.mMin.y - scroll + childSize.y}};
                            Arrange(childIndex, childRect, &childClip);
                        }
                    }
                },
                element.mBody);
    }

    void Ui::Impl::ArrangeLinear(const UiNode& node, const Rect& content,
            const std::vector<Element>& children, bool horizontal, Justify justify,
            Alignment align, float gap, const Rect* clip) {
        const uint32_t count = static_cast<uint32_t>(node.mChildren.size());
        if (count == 0) {
            return;
        }

        // A Spacer carries its size request in SpacerData (the Element's own
        // width/height stay Fit), so main-axis Grow must read it from there.
        const auto mainSizeOf = [](const Element& element, bool horizontal) -> Size {
            if (const SpacerData* spacer = std::get_if<SpacerData>(&element.mBody)) {
                return spacer->mSize;
            }
            return horizontal ? element.mWidth : element.mHeight;
        };

        std::vector<glm::vec2> sizes(count);
        float totalMain = 0.0f;
        float totalWeight = 0.0f;
        for (uint32_t i = 0; i < count; ++i) {
            const Element& child = *mNodes[node.mChildren[i]].mElement;
            sizes[i] = Measure(child, mNodes[node.mChildren[i]].mStyle, content.Size());
            const Size mainSize = mainSizeOf(child, horizontal);
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
            const Size mainSize = mainSizeOf(child, horizontal);
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
            Arrange(node.mChildren[i], childRect, clip);
            cursor += main + gap + extraGap;
        }
    }
}// namespace moe::ui
