// Font preprocessing smoke test: loads a TrueType font through the text
// pipeline and checks the packed glyph data (bands, curves, layout).

#include <Core/Defer.hpp>
#include <Core/Error.hpp>
#include <Neo/Assets.hpp>
#include <Neo/Font.hpp>
#include <RHI/Device.hpp>
#include <RHI/PipelineCache.hpp>

#include "TestSupport.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond)                                        \
    do {                                                   \
        if (!(cond)) {                                     \
            std::fprintf(stderr, "Font FAILED: %s (%d)\n", \
                    #cond, __LINE__);                      \
            return EXIT_FAILURE;                           \
        }                                                  \
    } while (false)

namespace {
    // Every band meta must point inside the glyph's own band block, and every
    // curve index must reference one of the glyph's curves.
    bool CheckBands(const moe::neo::FontData& font) {
        for (const moe::neo::Glyph& glyph : font.mGlyphs) {
            const uint32_t blockSize =
                    static_cast<uint32_t>(font.mBands.size()) - glyph.mBandOffset;
            for (uint32_t meta = 0; meta < moe::neo::kTextBandSplits * 2; ++meta) {
                const uint32_t dataOffset = font.mBands[glyph.mBandOffset + meta * 2];
                const uint32_t count = font.mBands[glyph.mBandOffset + meta * 2 + 1];
                if (dataOffset + count > blockSize) {
                    return false;
                }
                for (uint32_t i = 0; i < count; ++i) {
                    if (font.mBands[glyph.mBandOffset + dataOffset + i]
                            >= glyph.mCurves.size()) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // The control point of a curved segment must be off the chord; a straight
    // line (or a cubic mistaken for one) has the control on it.
    bool HasCurvature(const moe::neo::Glyph& glyph) {
        for (const moe::neo::GlyphCurve& curve : glyph.mCurves) {
            const glm::vec2 chord = curve.mTo - curve.mFrom;
            const float length = glm::length(chord);
            if (length < 1.0f) {
                continue;
            }
            const glm::vec2 relative = curve.mControl - curve.mFrom;
            const float distance = std::abs(chord.x * relative.y - chord.y * relative.x) / length;
            if (distance > length * 0.05f) {
                return true;
            }
        }
        return false;
    }
}// namespace

int main() {
    constexpr const char* kTestName = "Font smoke";
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }
    moe::Scheduler scheduler;
    moe::neo::TransferManager transfer;
    moe::neo::Assets assets;
    moe::Defer cleanup([&] {
        transfer.Shutdown();
        scheduler.Shutdown();
        assets.Destroy();
        cache.Destroy();
        device.Destroy();
    });
    if (!scheduler.Init(2) || !transfer.Init(device, scheduler)
            || !assets.Init(device, transfer)) {
        return moe::test::Fail(kTestName);
    }

    const char* kFont = MOE_SOURCE_DIR "/vendors/imgui/misc/fonts/Roboto-Medium.ttf";
    moe::neo::Font font = assets.LoadFont(kFont, "Hello, Text!");
    if (!font.IsValid()) {
        return moe::test::Fail(kTestName);
    }
    moe::neo::FontData* data = font.GetData();
    CHECK(data != nullptr);
    CHECK(data->mGlyphs.size() >= 95); // ASCII 32..126
    CHECK(data->mAscent > 0.0f);
    CHECK(data->mDescent < 0.0f);
    CHECK(data->GetScaleForPixelHeight(32.0f) > 0.0f);
    CHECK(CheckBands(*data));

    const moe::neo::Glyph* capitalH = data->FindGlyph(U'H');
    CHECK(capitalH != nullptr);
    CHECK(capitalH->mAdvance > 0.0f);
    CHECK(!capitalH->mCurves.empty());
    CHECK(HasCurvature(*data->FindGlyph(U'O')));

    moe::neo::TextDrawParams params;
    params.mPixelSize = 24.0f;
    const uint32_t vertexCount = data->BuildVertices("Hello\nText", params);
    CHECK(vertexCount > 0);
    CHECK(vertexCount % 6 == 0); // two triangles per glyph
    const uint32_t curveCount = static_cast<uint32_t>(data->mCurves.size() / 8);
    for (const moe::neo::TextVertex& vertex : data->mScratch) {
        CHECK(vertex.mTex[2] < static_cast<float>(curveCount));   // curve base
        CHECK(vertex.mTex[3] < static_cast<float>(data->mBands.size())); // band base
    }

    return EXIT_SUCCESS;
}
