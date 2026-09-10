#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe::neo {
    // CPU-side image data, format-agnostic (channels + bytes). The GPU uploader
    // picks the concrete RHI format. With mMipLevels > 1, mData holds the whole
    // chain tightly packed: level 0 first, then each level at half extent.
    struct Texture {
        std::string mName;
        uint32_t mWidth{0};
        uint32_t mHeight{0};
        uint32_t mDepth{1};
        uint32_t mMipLevels{1};
        uint32_t mChannels{4}; // 1, 2, 3 or 4
        bool mSrgb{false};     // color data (base color / emissive), not data
        std::vector<uint8_t> mData;
    };
}// namespace moe::neo