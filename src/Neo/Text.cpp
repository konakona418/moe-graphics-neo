// Text drawing from glyph outlines: builds glyph quads for a UTF-8 string,
// appends them to the frame's dynamic vertex arena and records one draw
// with the text shader.

#include "Neo/Renderer.hpp"

#include "Neo/Font.hpp"

#include <Core/Error.hpp>

#include <cstddef>

namespace moe::neo {
    namespace {
        const rhi::VertexAttribute kTextAttributes[5] = {
                {0, 0, rhi::Format::kR32G32B32A32Float, offsetof(TextVertex, mPos)},
                {1, 0, rhi::Format::kR32G32B32A32Float, offsetof(TextVertex, mTex)},
                {2, 0, rhi::Format::kR32G32B32A32Float, offsetof(TextVertex, mJac)},
                {3, 0, rhi::Format::kR32G32B32A32Float, offsetof(TextVertex, mBand)},
                {4, 0, rhi::Format::kR32G32B32A32Float, offsetof(TextVertex, mColor)},
        };

        // Descriptor binding lookup by SPIR-V name (set 0, any stage).
        const rhi::DescriptorBindingInfo* FindBinding(const rhi::ShaderProgram& program,
                const char* name) {
            const rhi::ShaderStage stages[2] = {
                    rhi::ShaderStage::kVertex, rhi::ShaderStage::kFragment};
            for (const rhi::ShaderStage stage : stages) {
                const rhi::Shader* shader = program.GetStage(stage);
                if (shader == nullptr) {
                    continue;
                }
                const auto& sets = shader->GetReflection().mDescriptorSets;
                if (sets.empty()) {
                    continue;
                }
                for (const auto& binding : sets[0]) {
                    if (binding.mName == name) {
                        return &binding;
                    }
                }
            }
            return nullptr;
        }

        void FeedMat4(PassContext& pass, const rhi::ShaderProgram& program, const char* name,
                const glm::mat4& value) {
            const int32_t index = pass.GetPushConstant(program, name);
            if (index < 0 || pass.GetPushConstantSize(index) != sizeof(glm::mat4)) {
                return;
            }
            pass.SetPushConstant(index, &value, sizeof(value));
        }

        void FeedVec2(PassContext& pass, const rhi::ShaderProgram& program, const char* name,
                const glm::vec2& value) {
            const int32_t index = pass.GetPushConstant(program, name);
            if (index < 0 || pass.GetPushConstantSize(index) != sizeof(glm::vec2)) {
                return;
            }
            pass.SetPushConstant(index, &value, sizeof(value));
        }
    }// namespace

    void PassContext::DrawText(const Font& font, std::string_view text,
            const TextDrawParams& params, ProgramHandle programHandle) {
        if (!font.IsValid()) {
            moe::Error::Set("DrawText: invalid font");
            return;
        }
        FontData* data = font.mAssets->GetFont(font.mHandle);
        rhi::ShaderProgram* program = font.mAssets->GetProgram(programHandle);
        if (data == nullptr || program == nullptr) {
            moe::Error::Set("DrawText: font or program is stale");
            return;
        }

        const uint32_t vertexCount = data->BuildVertices(text, params);
        if (vertexCount == 0) {
            return;
        }
        const uint32_t bytes = vertexCount * static_cast<uint32_t>(sizeof(TextVertex));
        const uint32_t offset = mRenderer->AppendDynamicVertices(data->mScratch.data(), bytes);
        if (offset == UINT32_MAX) {
            moe::Error::Set("DrawText: " + moe::Error::Get());
            return;
        }

        const Camera* camera = mRenderer->GetCameraInternal();
        const glm::mat4 viewProj =
                camera != nullptr ? camera->mProj * camera->mView : glm::mat4(1.0f);
        FeedMat4(*this, *program, "viewProj", viewProj);
        FeedMat4(*this, *program, "model", params.mTransform);
        FeedVec2(*this, *program, "viewport", mRenderer->GetViewportSizeInternal());

        ClearTextureBindings();
        const rhi::DescriptorBindingInfo* curves = FindBinding(*program, "curveData");
        const rhi::DescriptorBindingInfo* bands = FindBinding(*program, "bandData");
        if (curves != nullptr) {
            BindBuffer(curves->mBinding, data->mCurveBuffer);
        }
        if (bands != nullptr) {
            BindBuffer(bands->mBinding, data->mBandBuffer);
        }

        DrawVertices(mRenderer->GetDynamicVertexBufferInternal(), vertexCount, kTextAttributes, 5,
                sizeof(TextVertex), *program, rhi::PrimitiveTopology::kTriangleList,
                offset / static_cast<uint32_t>(sizeof(TextVertex)));
    }
}// namespace moe::neo
