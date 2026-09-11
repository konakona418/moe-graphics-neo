#include <UI/Style.hpp>

namespace moe::ui {
    ResolvedStyle Resolve(const Theme& theme, const Style& style) {
        ResolvedStyle out;
        out.mTextColor = style.mTextColor.value_or(theme.mTextColor);
        out.mBackground = style.mBackground.value_or(theme.mBackground);
        out.mBorderColor = style.mBorderColor.value_or(theme.mBorderColor);
        out.mRadius = style.mRadius.value_or(theme.mRadius);
        out.mBorderWidth = style.mBorderWidth.value_or(theme.mBorderWidth);
        out.mPadding = style.mPadding.value_or(theme.mPadding);
        out.mGap = style.mGap.value_or(theme.mGap);
        out.mFontSize = style.mFontSize.value_or(theme.mFontSize);
        out.mFont = style.mFont.value_or(theme.mFont);
        return out;
    }
}// namespace moe::ui
