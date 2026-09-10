#pragma once

#include <Neo/Texture.hpp>

#include <cstddef>

namespace moe::neo {
    // Decodes an image (PNG/JPEG/BMP/…, stb_image) from memory into an RGBA8
    // Texture. srgb marks the data as color data (the GPU uploader picks the
    // sRGB format, so sampling decodes to linear in shaders).
    bool DecodeTexture(const uint8_t* data, size_t size, Texture& out, bool srgb);
}// namespace moe::neo
