#pragma once

#include <UI/Element.hpp>
#include <UI/Event.hpp>
#include <UI/Style.hpp>

#include <RHI/Image.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace moe::rhi {
    class Device;
}

namespace moe::neo {
    class Assets;
    class Renderer;
}// namespace moe::neo

namespace moe::ui {
    // Declarative UI system. Internally immediate mode: every frame the caller
    // describes a view tree (value types, rebuilt from scratch), the UI lays it
    // out, dispatches input and records one offscreen pass into its own render
    // target. The caller samples GetImage() to composite.
    //
    //   ui.BeginFrame(desc);
    //   ui.EndFrame(Column({ Label("hi"), Button("ok", cb) }));
    //   ui.Render(renderer);
    //   // composite ui.GetImage()
    class Ui {
    public:
        Ui();
        ~Ui();

        Ui(const Ui&) = delete;
        Ui& operator=(const Ui&) = delete;

        // Loads the UI shaders and the 1x1 white texture. The renderer and
        // device must outlive the UI; assets own the programs/textures/fonts.
        bool Init(neo::Assets& assets, neo::Renderer& renderer, rhi::Device& device);
        void Destroy();

        // Starts a frame: stores the frame description and clears per-frame
        // scratch (events, draw list, flattened tree).
        void BeginFrame(const UiFrameDesc& desc);

        // Lays out `root`, runs the dispatch pass (hover/press/click, callbacks
        // and the event list) and builds the draw list. No GPU work.
        void EndFrame(const Element& root);

        // Records the UI's offscreen pass into its render target. Call between
        // BeginFrame/EndFrame of the renderer, before the composite pass that
        // samples GetImage().
        void Render(neo::Renderer& renderer);

        // The render target's sampleable image (valid after Render).
        const rhi::Image& GetImage() const;

        // Interactions produced by the last EndFrame.
        const std::vector<UiEvent>& GetEvents() const;

        Theme& GetTheme();
        const Theme& GetTheme() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::ui
