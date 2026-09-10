#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace moe::rhi {
    class Device;
    class CommandList;
    class Swapchain;
}

namespace moe::ui {
    struct DebugUIImpl;

    // ImGui debug overlay (imgui_impl_glfw + imgui_impl_vulkan, dynamic
    // rendering). The App drives the frame: call BeginFrame() once per frame,
    // let the app record ImGui windows, then call Render() while a render pass
    // on the target color image is active (the App begins/ends it). Migrated
    // from the old engine's initImGUI (same descriptor pool spec, dynamic
    // rendering mode, keyboard nav).
    class DebugUI {
    public:
        DebugUI();
        ~DebugUI();

        DebugUI(const DebugUI&) = delete;
        DebugUI& operator=(const DebugUI&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the UI was
        // initialized but not destroyed (leak trap, same discipline as the RHI).
        void Destroy();

        // Initializes ImGui + both backends. Needs the present-capable device,
        // the swapchain (its color format drives the pipeline) and the GLFW
        // window handle. Call after the swapchain exists, destroy before the
        // device.
        bool Init(moe::rhi::Device& device, moe::rhi::Swapchain& swapchain,
                uintptr_t glfwWindow);

        // Starts a new ImGui frame (record UI windows after this, then Render).
        void BeginFrame(float deltaSeconds);

        // Ends the ImGui frame and records the draw commands. A render pass on
        // the given color image must be active (the App begins it).
        void Render(moe::rhi::CommandList& cmd);

    private:
        std::unique_ptr<DebugUIImpl> mImpl;
    };
}// namespace moe::ui
