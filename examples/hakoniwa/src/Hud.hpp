#pragma once

#include <Neo/Assets.hpp>
#include <Neo/Input.hpp>
#include <Neo/Renderer.hpp>
#include <RHI/Sampler.hpp>
#include <UI/Ui.hpp>

#include <cstdint>

// HUD module: a moe::ui declarative overlay (menu / controls / in-game HUD) on
// a tilted, pointer-parallax layer, composited over the post result. State is
// read from and written back to HudState.

namespace hakoniwa {
    struct HudState {
        bool mShowMenu{true};
        bool mShowControls{false};
        bool mPhotoMode{false};
        bool mAutoCycle{false};
        bool mResetCamera{false};
        bool mWalkMode{true};
        float mSunElevationDeg{55.0f};
        float mSunAzimuthDeg{40.0f};
    };

    class Hud {
    public:
        bool Init(moe::rhi::Device& device, moe::neo::Assets& assets,
                moe::neo::Renderer& renderer);
        void Record(moe::neo::Assets& assets, moe::neo::Renderer& renderer,
                const moe::neo::Input& input, HudState& state,
                moe::neo::RenderTargetHandle minimapTarget, uint32_t width, uint32_t height);
        void Destroy();

    private:
        moe::ui::Ui mUi;
        moe::neo::Font mFont;
        moe::neo::ProgramHandle mComposite;
        moe::rhi::Sampler mSampler;
    };
}// namespace hakoniwa
