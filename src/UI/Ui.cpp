#include "UiInternal.hpp"

#include <Core/Error.hpp>

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

namespace moe::ui {
    namespace {
        uint64_t HashId(uint64_t parent, uint32_t index, const std::string& key,
                uint32_t typeTag) {
            uint64_t hash = 1469598103934665603ull;
            const auto mix = [&hash](uint64_t value) {
                hash ^= value;
                hash *= 1099511628211ull;
            };
            mix(parent);
            mix(typeTag);
            if (!key.empty()) {
                for (unsigned char c : key) {
                    mix(c);
                }
                mix(0xffu);
            } else {
                mix(index);
            }
            return hash;
        }

        void EmitEvent(std::vector<UiEvent>& events, const UiNode& node, UiEventType type) {
            UiEvent event;
            event.mId = node.mId;
            event.mType = type;
            if (const auto* button = std::get_if<ButtonData>(&node.mElement->mBody)) {
                event.mAction = button->mAction;
            }
            events.push_back(std::move(event));
        }
    }// namespace

    Style ElementStyle(const Element& element) {
        return std::visit(
                [](const auto& body) -> Style {
                    using T = std::decay_t<decltype(body)>;
                    if constexpr (requires { body.mStyle; }) {
                        return body.mStyle;
                    } else {
                        Style style;
                        style.mPadding = Insets::All(0.0f);
                        return style;
                    }
                },
                element.mBody);
    }

    bool IsInteractive(const Element& element) {
        return std::holds_alternative<ButtonData>(element.mBody);
    }

    Ui::Ui()
        : mImpl(std::make_unique<Impl>()) {}

    Ui::~Ui() {
        Destroy();
    }

    bool Ui::Impl::EnsureResources() {
        if (!mProgram.IsValid()) {
            mProgram = mAssets->LoadGraphicsProgram(MOE_SOURCE_DIR "/shaders/examples/ui.vert.spv",
                    MOE_SOURCE_DIR "/shaders/examples/ui.frag.spv");
            if (!mProgram.IsValid()) {
                return false;
            }
        }
        if (!mTextProgram.IsValid()) {
            mTextProgram = mAssets->LoadGraphicsProgram(
                    MOE_SOURCE_DIR "/shaders/examples/text.vert.spv",
                    MOE_SOURCE_DIR "/shaders/examples/text.frag.spv");
            if (!mTextProgram.IsValid()) {
                return false;
            }
        }
        if (!mWhite.IsValid()) {
            neo::Texture white;
            white.mWidth = 1;
            white.mHeight = 1;
            white.mChannels = 4;
            white.mData = {255, 255, 255, 255};
            mWhite = mAssets->UploadTexture(white);
            if (!mWhite.IsValid()) {
                return false;
            }
        }
        return true;
    }

    bool Ui::Impl::EnsureVertexCapacity(uint32_t vertexCount) {
        if (mVertexMapped != nullptr && vertexCount <= mVertexCapacity) {
            return true;
        }
        const uint32_t capacity = std::max(vertexCount, 4096u);
        if (mVertexMapped != nullptr) {
            mVertexBuffer.Unmap();
            mVertexMapped = nullptr;
        }
        mVertexBuffer.Destroy();
        rhi::BufferCreateInfo info{};
        info.mSize = static_cast<uint64_t>(capacity) * sizeof(UiVertex);
        info.mUsage = rhi::BufferUsage::kVertex;
        info.mCpuVisible = true;
        if (!mDevice->CreateBuffer(info, mVertexBuffer)) {
            return false;
        }
        mVertexMapped = mVertexBuffer.Map();
        mVertexCapacity = capacity;
        return mVertexMapped != nullptr;
    }

    bool Ui::Impl::EnsureTarget(uint32_t width, uint32_t height) {
        if (width == 0 || height == 0) {
            return false;
        }
        if (mTarget.IsValid() && mTargetWidth == width && mTargetHeight == height) {
            return true;
        }
        if (mTarget.IsValid()) {
            mRenderer->DestroyRenderTarget(mTarget);
            mTarget = {};
        }
        mTarget = mRenderer->CreateRenderTarget(width, height, rhi::Format::kR8G8B8A8Unorm,
                true, 0, true);
        if (!mTarget.IsValid()) {
            return false;
        }
        mTargetWidth = width;
        mTargetHeight = height;
        return true;
    }

    bool Ui::Init(neo::Assets& assets, neo::Renderer& renderer, rhi::Device& device) {
        Impl& impl = *mImpl;
        impl.mAssets = &assets;
        impl.mRenderer = &renderer;
        impl.mDevice = &device;
        return impl.EnsureResources();
    }

    void Ui::Destroy() {
        Impl& impl = *mImpl;
        if (impl.mVertexMapped != nullptr) {
            impl.mVertexBuffer.Unmap();
            impl.mVertexMapped = nullptr;
        }
        impl.mVertexBuffer.Destroy();
        if (impl.mRenderer != nullptr && impl.mTarget.IsValid()) {
            impl.mRenderer->DestroyRenderTarget(impl.mTarget);
            impl.mTarget = {};
        }
    }

    void Ui::Impl::ResetFrame() {
        mEvents.clear();
        mNodes.clear();
        mVertices.clear();
        mBatches.clear();
        mTexts.clear();
        mCommands.clear();
    }

    void Ui::BeginFrame(const UiFrameDesc& desc) {
        mImpl->mFrame = desc;
        mImpl->ResetFrame();
    }

    void Ui::Impl::Flatten(const Element& element, int32_t parent, uint64_t parentId,
            uint32_t childIndex) {
        if (!element.mVisible) {
            return;
        }
        const uint32_t index = static_cast<uint32_t>(mNodes.size());
        mNodes.emplace_back();
        const uint64_t id = HashId(parentId, childIndex, element.mKey,
                static_cast<uint32_t>(element.mBody.index()));
        {
            UiNode& node = mNodes[index];
            node.mElement = &element;
            node.mStyle = Resolve(mTheme, ElementStyle(element));
            node.mId = id;
            node.mParent = parent;
        }
        if (parent >= 0) {
            mNodes[parent].mChildren.push_back(index);
        }
        uint32_t next = 0;
        ForEachChild(element, [&](const Element& child) {
            Flatten(child, static_cast<int32_t>(index), id, next++);
        });
    }

    void Ui::Impl::Dispatch() {
        mEvents.clear();
        for (UiNode& node : mNodes) {
            node.mHovered = false;
        }

        // Wheel input: the topmost scroll view under the pointer consumes it.
        // The new offset is applied by the next frame's arrange (one-frame
        // latency, like every other input-driven layout change).
        if (!mFrame.mInput.mCaptured && mFrame.mInput.mScroll != 0.0f) {
            for (auto it = mNodes.rbegin(); it != mNodes.rend(); ++it) {
                UiNode& node = *it;
                if (!std::holds_alternative<ScrollViewData>(node.mElement->mBody)) {
                    continue;
                }
                const Rect viewport = node.mRect.Inset(node.mStyle.mPadding)
                        .Offset(mFrame.mOffset * (1.0f + node.mElement->mZ));
                if (!viewport.Contains(mFrame.mInput.mPointer)) {
                    continue;
                }
                if (node.mHasClip && !node.mClip.Contains(mFrame.mInput.mPointer)) {
                    continue;
                }
                mScrollState[node.mId] -= mFrame.mInput.mScroll * mTheme.mScrollStep;
                break;
            }
        }

        uint64_t hoveredId = 0;
        if (!mFrame.mInput.mCaptured) {
            for (auto it = mNodes.rbegin(); it != mNodes.rend(); ++it) {
                UiNode& node = *it;
                if (!node.mElement->mEnabled || !IsInteractive(*node.mElement)) {
                    continue;
                }
                // A point clipped away by any ancestor must not hit, even if it
                // is inside the element's own (unclipped) rectangle.
                if (node.mHasClip && !node.mClip.Contains(mFrame.mInput.mPointer)) {
                    continue;
                }
                const Rect hitRect = node.mRect.Offset(
                        mFrame.mOffset * (1.0f + node.mElement->mZ));
                if (hitRect.Contains(mFrame.mInput.mPointer)) {
                    hoveredId = node.mId;
                    break;
                }
            }
        }

        for (UiNode& node : mNodes) {
            if (!node.mElement->mEnabled || !IsInteractive(*node.mElement)) {
                continue;
            }
            const bool hovered = node.mId == hoveredId;
            node.mHovered = hovered;
            bool& previous = mHoverState[node.mId];
            if (hovered != previous) {
                EmitEvent(mEvents, node, hovered ? UiEventType::kHoverEnter
                                                 : UiEventType::kHoverLeave);
                previous = hovered;
            }
            if (hovered && mFrame.mInput.mPrimaryPressed) {
                mActiveId = node.mId;
                EmitEvent(mEvents, node, UiEventType::kPress);
            }
        }

        if (mFrame.mInput.mPrimaryReleased && mActiveId != 0) {
            for (UiNode& node : mNodes) {
                if (node.mId != mActiveId) {
                    continue;
                }
                EmitEvent(mEvents, node, UiEventType::kRelease);
                if (mActiveId == hoveredId) {
                    EmitEvent(mEvents, node, UiEventType::kClick);
                    if (auto* button = std::get_if<ButtonData>(&node.mElement->mBody)) {
                        if (button->mOnClick) {
                            button->mOnClick();
                        }
                    }
                }
                break;
            }
            mActiveId = 0;
        }
    }

    void Ui::EndFrame(const Element& root) {
        Impl& impl = *mImpl;
        impl.Flatten(root, -1, 0, 0);
        if (!impl.mNodes.empty()) {
            const float scale = impl.mFrame.mScale > 0.0f ? impl.mFrame.mScale : 1.0f;
            const float width = static_cast<float>(impl.mFrame.mWidth) / scale;
            const float height = static_cast<float>(impl.mFrame.mHeight) / scale;
            impl.Arrange(0, Rect{{0.0f, 0.0f}, {width, height}});
        }
        impl.Dispatch();
        impl.BuildDrawList();
    }

    void Ui::Render(neo::Renderer& renderer) {
        Impl& impl = *mImpl;
        if (!impl.EnsureTarget(impl.mFrame.mWidth, impl.mFrame.mHeight)) {
            return;
        }
        impl.Record(renderer);
    }

    const rhi::Image& Ui::GetImage() const {
        return *mImpl->mRenderer->GetRenderTarget(mImpl->mTarget)->mImage;
    }

    const std::vector<UiEvent>& Ui::GetEvents() const {
        return mImpl->mEvents;
    }

    Theme& Ui::GetTheme() {
        return mImpl->mTheme;
    }

    const Theme& Ui::GetTheme() const {
        return mImpl->mTheme;
    }
}// namespace moe::ui
