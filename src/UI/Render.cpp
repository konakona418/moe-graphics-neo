#include "UiInternal.hpp"

#include <Neo/Font.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace moe::ui {
    namespace {
        const rhi::VertexAttribute kUiAttributes[5] = {
                {0, 0, rhi::Format::kR32G32Float, offsetof(UiVertex, mPos)},
                {1, 0, rhi::Format::kR32G32Float, offsetof(UiVertex, mUv)},
                {2, 0, rhi::Format::kR32G32B32A32Float, offsetof(UiVertex, mColor)},
                {3, 0, rhi::Format::kR32G32B32A32Float, offsetof(UiVertex, mRect)},
                {4, 0, rhi::Format::kR32G32B32A32Float, offsetof(UiVertex, mParams)},
        };

        bool SameTexture(const neo::TextureHandle& a, const neo::TextureHandle& b) {
            return a.mIndex == b.mIndex && a.mGeneration == b.mGeneration;
        }
    }// namespace

    void Ui::Impl::BuildDrawList() {
        mVertices.clear();
        mBatches.clear();
        mTexts.clear();
        mCommands.clear();

        for (const UiNode& node : mNodes) {
            const Element& element = *node.mElement;
            const ResolvedStyle& style = node.mStyle;
            std::visit(
                    [&](const auto& body) {
                        using T = std::decay_t<decltype(body)>;
                        if constexpr (std::is_same_v<T, LabelData>) {
                            PushText(node.mRect, style, body.mText, body.mAlign, element.mZ);
                        } else if constexpr (std::is_same_v<T, ImageData>) {
                            PushRect(node.mRect, body.mTint, 0.0f, 0.0f, 0, element.mZ,
                                    body.mTexture);
                        } else if constexpr (std::is_same_v<T, ButtonData>) {
                            glm::vec4 background = mTheme.mButtonColor;
                            if (node.mHovered) {
                                background = mTheme.mButtonHover;
                            }
                            if (mActiveId == node.mId && node.mHovered) {
                                background = mTheme.mButtonActive;
                            }
                            if (body.mStyle.mBackground) {
                                background = *body.mStyle.mBackground;
                            }
                            PushRect(node.mRect, background, style.mRadius, style.mBorderWidth, 1,
                                    element.mZ, mWhite);
                            if (style.mBorderWidth > 0.0f) {
                                PushRect(node.mRect, style.mBorderColor, style.mRadius,
                                        style.mBorderWidth, 2, element.mZ, mWhite);
                            }
                            ResolvedStyle textStyle = style;
                            if (!body.mStyle.mTextColor) {
                                textStyle.mTextColor = mTheme.mButtonText;
                            }
                            PushText(node.mRect.Inset(style.mPadding), textStyle, body.mText,
                                    Alignment::kCenter, element.mZ);
                        } else if constexpr (std::is_same_v<T, PanelData>) {
                            PushRect(node.mRect, style.mBackground, style.mRadius,
                                    style.mBorderWidth, 1, element.mZ, mWhite);
                            if (style.mBorderWidth > 0.0f) {
                                PushRect(node.mRect, style.mBorderColor, style.mRadius,
                                        style.mBorderWidth, 2, element.mZ, mWhite);
                            }
                        }
                    },
                    element.mBody);
        }
    }

    void Ui::Impl::PushRect(const Rect& rect, const glm::vec4& color, float radius,
            float borderWidth, int mode, float z, neo::TextureHandle texture) {
        if (rect.Width() <= 0.0f || rect.Height() <= 0.0f) {
            return;
        }
        const uint32_t first = static_cast<uint32_t>(mVertices.size());
        const glm::vec2 center = rect.Center();
        const glm::vec2 half = rect.Size() * 0.5f;
        const glm::vec4 rectData{center.x, center.y, half.x, half.y};
        const glm::vec4 params{static_cast<float>(mode), radius, borderWidth, 1.0f + z};
        const glm::vec2 corners[4] = {
                rect.mMin, {rect.mMax.x, rect.mMin.y}, rect.mMax, {rect.mMin.x, rect.mMax.y}};
        const glm::vec2 uvs[4] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
        const int order[6] = {0, 1, 2, 0, 2, 3};
        for (const int k : order) {
            mVertices.push_back({corners[k], uvs[k], color, rectData, params});
        }
        if (!mCommands.empty() && mCommands.back().mType == UiCmdType::kRects) {
            UiRectBatch& last = mBatches[mCommands.back().mIndex];
            if (last.mFirstVertex + last.mVertexCount == first
                    && SameTexture(last.mTexture, texture)) {
                last.mVertexCount += 6;
                return;
            }
        }
        mBatches.push_back({texture, first, 6});
        mCommands.push_back({UiCmdType::kRects, static_cast<uint32_t>(mBatches.size() - 1)});
    }

    void Ui::Impl::PushText(const Rect& box, const ResolvedStyle& style, std::string_view text,
            Alignment align, float z) {
        if (text.empty() || !style.mFont.IsValid()) {
            return;
        }
        const neo::FontData* font = style.mFont.GetData();
        if (font == nullptr) {
            return;
        }
        const glm::vec2 size = MeasureText(*font, text, style.mFontSize);
        float x = box.mMin.x;
        if (align == Alignment::kCenter) {
            x = box.mMin.x + (box.Width() - size.x) * 0.5f;
        } else if (align == Alignment::kEnd) {
            x = box.mMax.x - size.x;
        }
        const float y = box.mMin.y + (box.Height() - size.y) * 0.5f;

        UiTextCmd command;
        command.mFont = style.mFont;
        command.mText = std::string(text);
        command.mColor = style.mTextColor;
        command.mFontSize = style.mFontSize;
        command.mPos = {x, y};
        command.mZ = z;
        mTexts.push_back(std::move(command));
        mCommands.push_back({UiCmdType::kText, static_cast<uint32_t>(mTexts.size() - 1)});
    }

    void Ui::Impl::Record(neo::Renderer& renderer) {
        if (!mVertices.empty()) {
            if (!EnsureVertexCapacity(static_cast<uint32_t>(mVertices.size()))) {
                return;
            }
            std::memcpy(mVertexMapped, mVertices.data(), mVertices.size() * sizeof(UiVertex));
        }

        rhi::ShaderProgram* program = mAssets->GetProgram(mProgram);
        if (program == nullptr) {
            return;
        }

        const float scale = mFrame.mScale > 0.0f ? mFrame.mScale : 1.0f;
        const float logicalWidth = static_cast<float>(mFrame.mWidth) / scale;
        const float logicalHeight = static_cast<float>(mFrame.mHeight) / scale;

        const neo::PassDesc pass{"ui", neo::ColorAttachment(mTarget, rhi::LoadOp::kClear), {}};
        renderer.Execute(pass, [&](neo::PassContext& context) {
            neo::DrawState state;
            state.mDepthTest = false;
            state.mDepthWrite = false;
            state.mCullMode = rhi::CullMode::kNone;
            state.mBlendEnabled = true;
            state.mBlendSrcColor = rhi::BlendFactor::kOne;
            state.mBlendDstColor = rhi::BlendFactor::kOneMinusSrcAlpha;
            state.mBlendSrcAlpha = rhi::BlendFactor::kOne;
            state.mBlendDstAlpha = rhi::BlendFactor::kOneMinusSrcAlpha;
            context.SetState(state);

            const glm::mat4 viewProj = mFrame.mViewProjection.value_or(
                    glm::ortho(0.0f, logicalWidth, 0.0f, logicalHeight, -1.0f, 1.0f));

            neo::Camera camera;
            camera.mProj = viewProj;
            camera.mView = glm::mat4(1.0f);
            context.SetCamera(camera);

            const int32_t pcViewProj = context.GetPushConstant(*program, "viewProj");
            const int32_t pcOffset = context.GetPushConstant(*program, "offset");

            for (const UiCmd& command : mCommands) {
                if (command.mType == UiCmdType::kRects) {
                    const UiRectBatch& batch = mBatches[command.mIndex];
                    neo::UploadedTexture* texture = mAssets->GetTexture(batch.mTexture);
                    if (texture == nullptr) {
                        continue;
                    }
                    context.ClearTextureBindings();
                    context.BindImage(0, texture->mImage);
                    context.BindSampler(1, texture->mSampler);
                    if (pcViewProj >= 0) {
                        context.SetPushConstant(pcViewProj, &viewProj, sizeof(viewProj));
                    }
                    if (pcOffset >= 0) {
                        context.SetPushConstant(pcOffset, &mFrame.mOffset,
                                sizeof(glm::vec2));
                    }
                    context.DrawVertices(mVertexBuffer, batch.mVertexCount, kUiAttributes, 5,
                            sizeof(UiVertex), *program, rhi::PrimitiveTopology::kTriangleList,
                            batch.mFirstVertex);
                    continue;
                }

                const UiTextCmd& text = mTexts[command.mIndex];
                neo::TextDrawParams params;
                params.mPixelSize = text.mFontSize;
                params.mColor = text.mColor;
                params.mTransform = glm::translate(glm::mat4(1.0f),
                        glm::vec3(text.mPos + mFrame.mOffset * (1.0f + text.mZ), 0.0f));
                context.DrawText(text.mFont, text.mText, params, mTextProgram);
            }
        });
    }
}// namespace moe::ui
