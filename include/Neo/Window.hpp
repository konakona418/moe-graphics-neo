#pragma once

#include <RHI/Device.hpp>

#include <cstdint>
#include <string>

namespace moe::neo {
    // GLFW window + a Vulkan surface created from the RHI device's instance.
    // Owns the surface; destroy before the device. Create/Destroy pair with a
    // leak-trap destructor (same discipline as the RHI).
    class Window {
    public:
        Window() = default;
        ~Window();

        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;

        void Destroy();

        // Creates the GLFW window and a surface on the device's instance.
        bool Create(moe::rhi::Device& device, uint32_t width, uint32_t height,
                const char* title);

        bool ShouldClose() const;
        void PollEvents();

        // Opaque VkSurfaceKHR handle for Device::CreateSwapchain.
        uintptr_t GetSurfaceHandle() const;

        // Opaque GLFWwindow* handle (for integrations that need it, e.g. ImGui).
        uintptr_t GetHandle() const;

    private:
        struct Impl;
        Impl* mImpl{nullptr};
    };
}// namespace moe::neo