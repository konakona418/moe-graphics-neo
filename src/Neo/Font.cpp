// Font preprocessing and layout for GPU glyph-outline text rendering
// (Eric Lengyel's "GPU-Centered Font Rendering Directly from Glyph
// Outlines").
//
// TrueType outlines are read with stb_truetype, converted to quadratic
// Bezier curves, and bucketed into horizontal/vertical bands so the text
// fragment shader only tests the curves near a sample. The packed data
// lives in two storage buffers; glyph quads are built on the CPU in text
// space (pixels, y down) and dilated per-vertex on the GPU, which keeps
// edges resolution independent at any transform.

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#include <Core/Profile.hpp>

#include "Neo/Font.hpp"

#include "Neo/Assets.hpp"

#include <Core/Error.hpp>
#include <Core/FileIo.hpp>
#include <Core/Logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace moe::neo {
    namespace {
        // Decodes one UTF-8 codepoint. Invalid bytes decode as U+FFFD and
        // advance one byte, so layout always makes progress.
        uint32_t DecodeUtf8(std::string_view text, size_t& index) {
            const auto byte = [&](size_t i) {
                return static_cast<uint8_t>(text[i]);
            };
            const uint8_t first = byte(index);
            if (first < 0x80) {
                index += 1;
                return first;
            }
            uint32_t codepoint = 0xFFFD;
            size_t length = 1;
            if ((first & 0xE0) == 0xC0) {
                codepoint = first & 0x1F;
                length = 2;
            } else if ((first & 0xF0) == 0xE0) {
                codepoint = first & 0x0F;
                length = 3;
            } else if ((first & 0xF8) == 0xF0) {
                codepoint = first & 0x07;
                length = 4;
            }
            if (index + length > text.size()) {
                index += 1;
                return 0xFFFD;
            }
            for (size_t i = 1; i < length; ++i) {
                const uint8_t continuation = byte(index + i);
                if ((continuation & 0xC0) != 0x80) {
                    index += 1;
                    return 0xFFFD;
                }
                codepoint = (codepoint << 6) | (continuation & 0x3F);
            }
            index += length;
            return codepoint;
        }

        float Min3(float a, float b, float c) {
            return std::min(a, std::min(b, c));
        }

        float Max3(float a, float b, float c) {
            return std::max(a, std::max(b, c));
        }

        // The coverage solver only handles quadratic curves, but CFF/
        // PostScript fonts (e.g. the Noto CJK collections) emit cubic
        // segments. A cubic is approximated by the quadratic sharing its
        // endpoints and tangents (control = 0.75 c1 + 0.75 c2 - 0.25 p0 -
        // 0.25 p3); when the midpoint deviates more than the tolerance, the
        // cubic is split at t = 0.5 and both halves are converted.
        void AppendCubicQuadratics(const glm::vec2& p0, const glm::vec2& c1, const glm::vec2& c2,
                const glm::vec2& p3, std::vector<GlyphCurve>& out, int depth) {
            const glm::vec2 control = 0.75f * c1 + 0.75f * c2 - 0.25f * p0 - 0.25f * p3;
            const glm::vec2 cubicMid = 0.125f * p0 + 0.375f * c1 + 0.375f * c2 + 0.125f * p3;
            const glm::vec2 quadraticMid = 0.25f * p0 + 0.5f * control + 0.25f * p3;
            const float error = glm::length(cubicMid - quadraticMid);
            if (depth >= 6 || error < 0.25f) {
                out.push_back({p0, control, p3});
                return;
            }
            const glm::vec2 p01 = 0.5f * (p0 + c1);
            const glm::vec2 p12 = 0.5f * (c1 + c2);
            const glm::vec2 p23 = 0.5f * (c2 + p3);
            const glm::vec2 p012 = 0.5f * (p01 + p12);
            const glm::vec2 p123 = 0.5f * (p12 + p23);
            const glm::vec2 mid = 0.5f * (p012 + p123);
            AppendCubicQuadratics(p0, p01, p012, mid, out, depth + 1);
            AppendCubicQuadratics(mid, p123, p23, p3, out, depth + 1);
        }

        // Buckets a glyph's curves into the x/y band slabs. A curve belongs
        // to a slab when its bounding box overlaps the slab; within a slab
        // the curves are sorted by descending maximum on the ray axis, which
        // is what lets the shader stop early.
        void BuildBands(Glyph& glyph) {
            const float width = glyph.mBoundsMax.x - glyph.mBoundsMin.x;
            const float height = glyph.mBoundsMax.y - glyph.mBoundsMin.y;
            if (width <= 0.0f || height <= 0.0f || glyph.mCurves.empty()) {
                return;
            }
            struct Candidate {
                uint16_t mCurve;
                float mSortKey;
            };
            std::vector<Candidate> xCandidates;
            std::vector<Candidate> yCandidates;
            for (uint32_t slab = 0; slab < kTextBandSplits; ++slab) {
                const float xMin = glyph.mBoundsMin.x + width * slab / kTextBandSplits;
                const float xMax = glyph.mBoundsMin.x + width * (slab + 1) / kTextBandSplits;
                const float yMin = glyph.mBoundsMin.y + height * slab / kTextBandSplits;
                const float yMax = glyph.mBoundsMin.y + height * (slab + 1) / kTextBandSplits;

                xCandidates.clear();
                yCandidates.clear();
                for (uint32_t c = 0; c < glyph.mCurves.size(); ++c) {
                    const GlyphCurve& curve = glyph.mCurves[c];
                    const float curveX0 = Min3(curve.mFrom.x, curve.mControl.x, curve.mTo.x);
                    const float curveX1 = Max3(curve.mFrom.x, curve.mControl.x, curve.mTo.x);
                    const float curveY0 = Min3(curve.mFrom.y, curve.mControl.y, curve.mTo.y);
                    const float curveY1 = Max3(curve.mFrom.y, curve.mControl.y, curve.mTo.y);
                    if (curveX1 >= xMin && curveX0 <= xMax) {
                        xCandidates.push_back({static_cast<uint16_t>(c), curveY1});
                    }
                    if (curveY1 >= yMin && curveY0 <= yMax) {
                        yCandidates.push_back({static_cast<uint16_t>(c), curveX1});
                    }
                }
                const auto descending = [](const Candidate& a, const Candidate& b) {
                    return a.mSortKey > b.mSortKey;
                };
                std::sort(xCandidates.begin(), xCandidates.end(), descending);
                std::sort(yCandidates.begin(), yCandidates.end(), descending);
                for (const Candidate& candidate : xCandidates) {
                    glyph.mBandX[slab].push_back(candidate.mCurve);
                }
                for (const Candidate& candidate : yCandidates) {
                    glyph.mBandY[slab].push_back(candidate.mCurve);
                }
            }
        }

        // Appends one glyph's curves and band block to the font's packed
        // arrays. Band blocks are [x slab metas][y slab metas][x indices]
        // [y indices]; meta offsets are relative to the block start, matching
        // the text shader.
        void PackGlyph(FontData& font, Glyph& glyph) {
            glyph.mCurveOffset = static_cast<uint32_t>(font.mCurves.size() / 8);
            for (const GlyphCurve& curve : glyph.mCurves) {
                font.mCurves.push_back(curve.mFrom.x);
                font.mCurves.push_back(curve.mFrom.y);
                font.mCurves.push_back(curve.mControl.x);
                font.mCurves.push_back(curve.mControl.y);
                font.mCurves.push_back(curve.mTo.x);
                font.mCurves.push_back(curve.mTo.y);
                font.mCurves.push_back(0.0f);
                font.mCurves.push_back(0.0f);
            }

            constexpr uint32_t kMetaUnits = kTextBandSplits * 2 * 2;
            glyph.mBandOffset = static_cast<uint32_t>(font.mBands.size());
            font.mBands.resize(glyph.mBandOffset + kMetaUnits);

            uint32_t indexCount = 0;
            for (uint32_t slab = 0; slab < kTextBandSplits; ++slab) {
                font.mBands[glyph.mBandOffset + slab * 2] = kMetaUnits + indexCount;
                font.mBands[glyph.mBandOffset + slab * 2 + 1] =
                        static_cast<uint32_t>(glyph.mBandX[slab].size());
                indexCount += static_cast<uint32_t>(glyph.mBandX[slab].size());
            }
            for (uint32_t slab = 0; slab < kTextBandSplits; ++slab) {
                const uint32_t meta = kTextBandSplits * 2 + slab * 2;
                font.mBands[glyph.mBandOffset + meta] = kMetaUnits + indexCount;
                font.mBands[glyph.mBandOffset + meta + 1] =
                        static_cast<uint32_t>(glyph.mBandY[slab].size());
                indexCount += static_cast<uint32_t>(glyph.mBandY[slab].size());
            }
            for (uint32_t slab = 0; slab < kTextBandSplits; ++slab) {
                font.mBands.insert(font.mBands.end(), glyph.mBandX[slab].begin(),
                        glyph.mBandX[slab].end());
            }
            for (uint32_t slab = 0; slab < kTextBandSplits; ++slab) {
                font.mBands.insert(font.mBands.end(), glyph.mBandY[slab].begin(),
                        glyph.mBandY[slab].end());
            }
        }

        bool BuildFontData(const char* path, const std::vector<uint32_t>& codepoints,
                FontData& out) {
            std::vector<uint8_t> bytes;
            if (!moe::ReadFileBytes(path, bytes)) {
                return false;
            }
            const int offset = stbtt_GetFontOffsetForIndex(bytes.data(), 0);
            if (offset < 0) {
                return moe::Fail(std::string("Font: not a TrueType font: ") + path);
            }
            stbtt_fontinfo info{};
            if (!stbtt_InitFont(&info, bytes.data(), offset)) {
                return moe::Fail(std::string("Font: failed to parse: ") + path);
            }
            int ascent = 0;
            int descent = 0;
            int lineGap = 0;
            stbtt_GetFontVMetrics(&info, &ascent, &descent, &lineGap);
            out.mAscent = static_cast<float>(ascent);
            out.mDescent = static_cast<float>(descent);
            out.mLineGap = static_cast<float>(lineGap);

            for (const uint32_t codepoint : codepoints) {
                if (out.mCodepointToGlyph.find(codepoint) != out.mCodepointToGlyph.end()) {
                    continue;
                }
                Glyph glyph;
                glyph.mCodepoint = codepoint;
                int advance = 0;
                int leftSideBearing = 0;
                stbtt_GetCodepointHMetrics(&info, static_cast<int>(codepoint), &advance,
                        &leftSideBearing);
                glyph.mAdvance = static_cast<float>(advance);
                glyph.mLeftSideBearing = static_cast<float>(leftSideBearing);
                int x0 = 0;
                int y0 = 0;
                int x1 = 0;
                int y1 = 0;
                if (stbtt_GetCodepointBox(&info, static_cast<int>(codepoint), &x0, &y0, &x1, &y1)) {
                    glyph.mBoundsMin = {static_cast<float>(x0), static_cast<float>(y0)};
                    glyph.mBoundsMax = {static_cast<float>(x1), static_cast<float>(y1)};
                }

                stbtt_vertex* vertices = nullptr;
                const int vertexCount =
                        stbtt_GetCodepointShape(&info, static_cast<int>(codepoint), &vertices);
                int lastX = 0;
                int lastY = 0;
                for (int i = 0; i < vertexCount; ++i) {
                    const stbtt_vertex& vertex = vertices[i];
                    if (vertex.type == STBTT_vmove) {
                        lastX = vertex.x;
                        lastY = vertex.y;
                        continue;
                    }
                    GlyphCurve curve;
                    curve.mFrom = {static_cast<float>(lastX), static_cast<float>(lastY)};
                    if (vertex.type == STBTT_vcurve) {
                        curve.mControl = {static_cast<float>(vertex.cx),
                                static_cast<float>(vertex.cy)};
                        curve.mTo = {static_cast<float>(vertex.x), static_cast<float>(vertex.y)};
                        glyph.mCurves.push_back(curve);
                    } else if (vertex.type == STBTT_vcubic) {
                        // CFF outlines: convert the cubic to quadratics.
                        AppendCubicQuadratics(curve.mFrom,
                                {static_cast<float>(vertex.cx), static_cast<float>(vertex.cy)},
                                {static_cast<float>(vertex.cx1), static_cast<float>(vertex.cy1)},
                                {static_cast<float>(vertex.x), static_cast<float>(vertex.y)},
                                glyph.mCurves, 0);
                    } else {
                        // Straight segments are degenerate quadratics; the
                        // control point must stay on the segment.
                        curve.mControl = curve.mFrom;
                        curve.mTo = {static_cast<float>(vertex.x), static_cast<float>(vertex.y)};
                        glyph.mCurves.push_back(curve);
                    }
                    lastX = vertex.x;
                    lastY = vertex.y;
                }
                if (vertices != nullptr) {
                    stbtt_FreeShape(&info, vertices);
                }

                BuildBands(glyph);
                out.mCodepointToGlyph[codepoint] = static_cast<uint32_t>(out.mGlyphs.size());
                out.mGlyphs.push_back(std::move(glyph));
            }

            for (Glyph& glyph : out.mGlyphs) {
                PackGlyph(out, glyph);
            }

            for (const uint32_t first : codepoints) {
                for (const uint32_t second : codepoints) {
                    const int kern = stbtt_GetCodepointKernAdvance(&info,
                            static_cast<int>(first), static_cast<int>(second));
                    if (kern != 0) {
                        out.mKerning[(static_cast<uint64_t>(first) << 32) | second] =
                                static_cast<float>(kern);
                    }
                }
            }
            return true;
        }

        // Maps a glyph's font-space bounding box (y up) to a text-space quad
        // (pixels, y down) anchored at the pen and baseline, then emits the
        // two triangles of the quad.
        void EmitGlyph(const FontData& font, const Glyph& glyph, float penX, float baseline,
                float scale, const glm::vec4& color, std::vector<TextVertex>& out) {
            const float x0 = glyph.mBoundsMin.x;
            const float y0 = glyph.mBoundsMin.y;
            const float x1 = glyph.mBoundsMax.x;
            const float y1 = glyph.mBoundsMax.y;

            const float left = penX + x0 * scale;
            const float right = penX + x1 * scale;
            const float top = baseline - y1 * scale;
            const float bottom = baseline - y0 * scale;

            const glm::vec2 p0{left, top};
            const glm::vec2 p1{right, top};
            const glm::vec2 p3{left, bottom};
            const glm::vec2 t0{x0, y1};
            const glm::vec2 t1{x1, y1};
            const glm::vec2 t3{x0, y0};

            const glm::vec2 pDx = p1 - p0;
            const glm::vec2 pDy = p3 - p0;
            const glm::vec2 tDx = t1 - t0;
            const glm::vec2 tDy = t3 - t0;
            const float det = pDx.x * pDy.y - pDy.x * pDx.y;
            glm::vec4 jac{1.0f, 0.0f, 0.0f, 1.0f};
            if (std::abs(det) > 1.0e-8f) {
                const float invDet = 1.0f / det;
                const float invP00 = pDy.y * invDet;
                const float invP01 = -pDy.x * invDet;
                const float invP10 = -pDx.y * invDet;
                const float invP11 = pDx.x * invDet;
                jac = {
                        tDx.x * invP00 + tDy.x * invP10,
                        tDx.x * invP01 + tDy.x * invP11,
                        tDx.y * invP00 + tDy.y * invP10,
                        tDx.y * invP01 + tDy.y * invP11,
                };
            }

            const float width = x1 - x0;
            const float height = y1 - y0;
            const float bandScaleX = width > 0.0f ? kTextBandSplits / width : 0.0f;
            const float bandScaleY = height > 0.0f ? kTextBandSplits / height : 0.0f;
            const float bandOffsetX = -x0 * bandScaleX;
            const float bandOffsetY = -y0 * bandScaleY;

            const float curveBase = static_cast<float>(glyph.mCurveOffset);
            const float bandBase = static_cast<float>(glyph.mBandOffset);

            const auto makeVertex = [&](float x, float y, float nx, float ny, float u, float v) {
                TextVertex vertex{};
                vertex.mPos[0] = x;
                vertex.mPos[1] = y;
                vertex.mPos[2] = nx;
                vertex.mPos[3] = ny;
                vertex.mTex[0] = u;
                vertex.mTex[1] = v;
                vertex.mTex[2] = curveBase;
                vertex.mTex[3] = bandBase;
                vertex.mJac[0] = jac.x;
                vertex.mJac[1] = jac.y;
                vertex.mJac[2] = jac.z;
                vertex.mJac[3] = jac.w;
                vertex.mBand[0] = bandScaleX;
                vertex.mBand[1] = bandScaleY;
                vertex.mBand[2] = bandOffsetX;
                vertex.mBand[3] = bandOffsetY;
                vertex.mColor[0] = color.r;
                vertex.mColor[1] = color.g;
                vertex.mColor[2] = color.b;
                vertex.mColor[3] = color.a;
                return vertex;
            };

            const TextVertex topLeft = makeVertex(left, top, -1.0f, -1.0f, x0, y1);
            const TextVertex topRight = makeVertex(right, top, 1.0f, -1.0f, x1, y1);
            const TextVertex bottomRight = makeVertex(right, bottom, 1.0f, 1.0f, x1, y0);
            const TextVertex bottomLeft = makeVertex(left, bottom, -1.0f, 1.0f, x0, y0);
            out.push_back(topLeft);
            out.push_back(topRight);
            out.push_back(bottomRight);
            out.push_back(topLeft);
            out.push_back(bottomRight);
            out.push_back(bottomLeft);
        }
    }// namespace

    float FontData::GetScaleForPixelHeight(float pixelHeight) const {
        const float height = mAscent - mDescent;
        return height > 0.0f ? pixelHeight / height : 0.0f;
    }

    float FontData::GetLineAdvance() const {
        return mAscent - mDescent + mLineGap;
    }

    const Glyph* FontData::FindGlyph(uint32_t codepoint) const {
        const auto it = mCodepointToGlyph.find(codepoint);
        if (it == mCodepointToGlyph.end() || it->second >= mGlyphs.size()) {
            return nullptr;
        }
        return &mGlyphs[it->second];
    }

    float FontData::GetKernAdvance(uint32_t first, uint32_t second) const {
        const auto it = mKerning.find((static_cast<uint64_t>(first) << 32) | second);
        return it != mKerning.end() ? it->second : 0.0f;
    }

    uint32_t FontData::BuildVertices(std::string_view text, const TextDrawParams& params) {
        MOE_PROFILE_ZONE();
        mScratch.clear();
        const float scale = GetScaleForPixelHeight(params.mPixelSize);
        if (scale <= 0.0f) {
            return 0;
        }
        const float lineAdvance = GetLineAdvance() * scale + params.mLineSpacing;
        float penX = 0.0f;
        float baseline = mAscent * scale;
        uint32_t previous = 0;
        size_t index = 0;
        while (index < text.size()) {
            const uint32_t codepoint = DecodeUtf8(text, index);
            if (codepoint == '\n') {
                penX = 0.0f;
                baseline += lineAdvance;
                previous = 0;
                continue;
            }
            if (codepoint == '\r') {
                continue;
            }
            const Glyph* glyph = FindGlyph(codepoint);
            if (glyph == nullptr) {
                previous = 0;
                continue;
            }
            penX += GetKernAdvance(previous, codepoint) * scale;
            if (!glyph->mCurves.empty()) {
                EmitGlyph(*this, *glyph, penX, baseline, scale, params.mColor, mScratch);
            }
            penX += glyph->mAdvance * scale + params.mLetterSpacing;
            previous = codepoint;
        }
        return static_cast<uint32_t>(mScratch.size());
    }

    void FontData::Destroy() {
        MOE_PROFILE_ZONE();
        mCurveBuffer.Destroy();
        mBandBuffer.Destroy();
    }

    Font Assets::LoadFont(const char* path, std::string_view sampleText) {
        MOE_PROFILE_ZONE();
        if (mDevice == nullptr) {
            moe::Error::Set("Assets: not initialized");
            return {};
        }

        std::vector<uint32_t> codepoints;
        for (uint32_t codepoint = 32; codepoint < 127; ++codepoint) {
            codepoints.push_back(codepoint);
        }
        size_t index = 0;
        while (index < sampleText.size()) {
            const uint32_t codepoint = DecodeUtf8(sampleText, index);
            if (codepoint >= 32 && codepoint != 127) {
                codepoints.push_back(codepoint);
            }
        }
        std::sort(codepoints.begin(), codepoints.end());
        codepoints.erase(std::unique(codepoints.begin(), codepoints.end()), codepoints.end());

        const FontHandle handle = mFonts.Emplace();
        FontData* font = mFonts.Get(handle);
        font->mName = path;
        if (!BuildFontData(path, codepoints, *font)) {
            font->Destroy();
            mFonts.Remove(handle);
            return {};
        }
        if (!mTransfer->UploadData(reinterpret_cast<const uint8_t*>(font->mCurves.data()),
                    font->mCurves.size() * sizeof(float), rhi::BufferUsage::kStorage,
                    font->mCurveBuffer, rhi::PipelineStage::kFragmentShader)
                || !mTransfer->UploadData(reinterpret_cast<const uint8_t*>(font->mBands.data()),
                        font->mBands.size() * sizeof(uint32_t), rhi::BufferUsage::kStorage,
                        font->mBandBuffer, rhi::PipelineStage::kFragmentShader)) {
            font->Destroy();
            mFonts.Remove(handle);
            return {};
        }

        moe::Logger::Info("Loaded font '{}' ({} glyphs, {} curves, {} band uints)", path,
                font->mGlyphs.size(), font->mCurves.size() / 8, font->mBands.size());
        return Font(this, handle);
    }

    FontData* Assets::GetFont(FontHandle handle) {
        return mFonts.Get(handle);
    }

    FontData* Font::GetData() const {
        return mAssets != nullptr ? mAssets->GetFont(mHandle) : nullptr;
    }
}// namespace moe::neo
