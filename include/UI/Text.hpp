#pragma once

#include <Neo/Assets.hpp>

#include <glm/glm.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace moe::ui {
    // How a text element turns its string into lines.
    //   kSingleLine: only explicit '\n' breaks (the default).
    //   kWrap:       word-wrap to the available width; a word wider than the
    //                line is broken by character.
    //   kEllipsis:   one line, truncated to the available width with "...".
    enum class TextLayout { kSingleLine, kWrap, kEllipsis };

    // One measured line. mText has no trailing newline; mWidth is its advance.
    struct TextLine {
        std::string mText;
        float mWidth{0.0f};
    };

    // Verbose measurement result: the block size (max line width x total
    // height) plus the individual lines and the line height.
    struct TextMetrics {
        glm::vec2 mSize{0.0f};
        float mLineHeight{0.0f};
        std::vector<TextLine> mLines;
    };

    // Measures `text` with `font` at `pixelSize`, applying `layout` within
    // `maxWidth` logical pixels (ignored by kSingleLine; a non-positive
    // maxWidth also disables wrapping/truncation). Works on the CPU alone, so
    // callers can measure before a Ui or a device exists.
    TextMetrics MeasureText(const neo::Font& font, std::string_view text, float pixelSize,
            TextLayout layout = TextLayout::kSingleLine, float maxWidth = 0.0f);
}// namespace moe::ui
