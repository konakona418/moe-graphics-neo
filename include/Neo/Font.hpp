#pragma once

#include <Neo/Cache.hpp>

#include <RHI/Buffer.hpp>

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace moe::neo {
    class Assets;
    class PassContext;

    // Band count of the glyph band acceleration structure. The CPU packer
    // and the text shader must agree (kBandSplits in text.slang).
    inline constexpr uint32_t kTextBandSplits = 16;

    // One quadratic Bezier curve of a glyph outline, in font units.
    struct GlyphCurve {
        glm::vec2 mFrom{0.0f};
        glm::vec2 mControl{0.0f};
        glm::vec2 mTo{0.0f};
    };

    // One glyph: outline, metrics and the band acceleration structure the
    // text shader walks. Curves are in font units (y up); the layout code
    // maps them to text space (pixels, y down).
    struct Glyph {
        uint32_t mCodepoint{0};
        std::vector<GlyphCurve> mCurves;
        // Candidate curve indices per slab, sorted by descending maximum on
        // the ray axis (the shader early-outs on that order).
        std::vector<uint16_t> mBandX[kTextBandSplits]; // x slabs, horizontal rays
        std::vector<uint16_t> mBandY[kTextBandSplits]; // y slabs, vertical rays
        glm::vec2 mBoundsMin{0.0f}; // font units
        glm::vec2 mBoundsMax{0.0f};
        float mAdvance{0.0f};
        float mLeftSideBearing{0.0f};
        uint32_t mCurveOffset{0}; // first curve in FontData::mCurves
        uint32_t mBandOffset{0};  // first uint in FontData::mBands
    };

    // One text vertex: 5 vec4 attributes, matching text.slang.
    struct TextVertex {
        float mPos[4];   // object xy, object-space normal xy
        float mTex[4];   // em xy, curve base, band base (plain float values)
        float mJac[4];   // inverse Jacobian (00, 01, 10, 11)
        float mBand[4];  // band scale xy, band offset xy
        float mColor[4]; // straight (non-premultiplied) RGBA
    };

    // Text drawing parameters. The text is laid out in pixel space (y down,
    // origin at the block's top-left) and mapped through mTransform; the
    // text shader keeps the edges resolution independent, so the transform
    // may scale/rotate freely.
    struct TextDrawParams {
        glm::mat4 mTransform{1.0f};
        glm::vec4 mColor{1.0f, 1.0f, 1.0f, 1.0f};
        // Pixel height of (ascent - descent); the same convention as
        // stbtt_ScaleForPixelHeight.
        float mPixelSize{32.0f};
        float mLetterSpacing{0.0f}; // extra pixels between glyphs
        float mLineSpacing{0.0f};   // extra pixels between baselines
    };

    // CPU + GPU font asset: glyph outlines and bands, layout, and the GPU
    // buffers the text shader reads. Produced by Assets::LoadFont.
    struct FontData {
        std::string mName;
        std::vector<Glyph> mGlyphs;
        std::unordered_map<uint32_t, uint32_t> mCodepointToGlyph;
        // Nonzero kerning pairs of the loaded codepoint set, in font units,
        // keyed by (first << 32 | second).
        std::unordered_map<uint64_t, float> mKerning;
        float mAscent{0.0f};
        float mDescent{0.0f};
        float mLineGap{0.0f};

        // Packed glyph data: 8 floats per curve (from, control, to, padding),
        // and per-glyph band blocks (metas then curve index lists).
        std::vector<float> mCurves;
        std::vector<uint32_t> mBands;

        // GPU resources.
        rhi::Buffer mCurveBuffer;
        rhi::Buffer mBandBuffer;

        // Layout scratch, reused across draws.
        std::vector<TextVertex> mScratch;

        float GetScaleForPixelHeight(float pixelHeight) const;
        float GetLineAdvance() const;
        const Glyph* FindGlyph(uint32_t codepoint) const;
        float GetKernAdvance(uint32_t first, uint32_t second) const;

        // Lays out UTF-8 `text` and fills mScratch with triangle-list
        // vertices. Returns the vertex count.
        uint32_t BuildVertices(std::string_view text, const TextDrawParams& params);

        void Destroy();
    };

    using FontHandle = Handle<FontData>;

    // Lightweight value handle to a font. Configuration resolves through the
    // owning Assets; stale handles are inert.
    class Font {
    public:
        Font() = default;

        bool IsValid() const {
            return mAssets != nullptr && mHandle.IsValid();
        }

        // Resolved through the owning Assets; null for a stale handle.
        FontData* GetData() const;

    private:
        friend class Assets;
        friend class PassContext;

        Font(Assets* assets, FontHandle handle)
            : mAssets(assets)
            , mHandle(handle) {}

        Assets* mAssets{nullptr};
        FontHandle mHandle;
    };
}// namespace moe::neo
