#include <Neo/TextureLoader.hpp>

#include <Core/Logger.hpp>

#include <cstring>

// Single translation unit providing the stb_image implementation for moe-neo.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace moe::neo {
    bool DecodeTexture(const uint8_t* data, size_t size, Texture& out, bool srgb) {
        int width = 0;
        int height = 0;
        int channels = 0;
        uint8_t* pixels = stbi_load_from_memory(data, static_cast<int>(size),
                &width, &height, &channels, 4); // force RGBA8
        if (pixels == nullptr) {
            moe::Logger::error("[neo] texture: failed to decode image: {}",
                    stbi_failure_reason());
            return false;
        }

        out.mName.clear();
        out.mWidth = static_cast<uint32_t>(width);
        out.mHeight = static_cast<uint32_t>(height);
        out.mChannels = 4;
        out.mSrgb = srgb;
        const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        out.mData.assign(pixels, pixels + bytes);
        stbi_image_free(pixels);
        return true;
    }
}// namespace moe::neo
