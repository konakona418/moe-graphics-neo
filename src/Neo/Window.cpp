#include "Neo/Window.hpp"

// volk (Vulkan types) must be visible to glfw3.h for its Vulkan functions.
#include <volk.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>

namespace moe::neo {
    struct Window::Impl {
        GLFWwindow* mWindow{nullptr};
        VkSurfaceKHR mSurface{VK_NULL_HANDLE};
        VkInstance mInstance{VK_NULL_HANDLE};
    };

    Window::~Window() {
        if (mImpl != nullptr) {
            std::fprintf(stderr, "[neo] Window leaked: Destroy() not called\n");
            std::abort();
        }
    }

    void Window::Destroy() {
        if (mImpl == nullptr) {
            return;
        }
        if (mImpl->mSurface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(mImpl->mInstance, mImpl->mSurface, nullptr);
        }
        if (mImpl->mWindow != nullptr) {
            glfwDestroyWindow(mImpl->mWindow);
        }
        delete mImpl;
        mImpl = nullptr;
    }

    bool Window::Create(moe::rhi::Device& device, uint32_t width, uint32_t height,
            const char* title, std::string& error) {
        mImpl = new Impl();

        if (glfwInit() != GLFW_TRUE) {
            error = "glfwInit failed";
            delete mImpl;
            mImpl = nullptr;
            return false;
        }
        if (glfwVulkanSupported() != GLFW_TRUE) {
            error = "glfw: Vulkan not supported";
            delete mImpl;
            mImpl = nullptr;
            return false;
        }
        // tell GLFW to resolve instance functions through the same loader volk uses
        glfwInitVulkanLoader(vkGetInstanceProcAddr);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        mImpl->mWindow = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (mImpl->mWindow == nullptr) {
            error = "glfwCreateWindow failed";
            delete mImpl;
            mImpl = nullptr;
            return false;
        }

        uintptr_t instanceHandle = 0;
        if (!device.GetInstanceHandle(instanceHandle)) {
            error = "device has no instance (mEnablePresent required)";
            glfwDestroyWindow(mImpl->mWindow);
            delete mImpl;
            mImpl = nullptr;
            return false;
        }
        mImpl->mInstance = reinterpret_cast<VkInstance>(instanceHandle);

        if (glfwCreateWindowSurface(mImpl->mInstance, mImpl->mWindow, nullptr, &mImpl->mSurface) != VK_SUCCESS) {
            error = "glfwCreateWindowSurface failed";
            glfwDestroyWindow(mImpl->mWindow);
            delete mImpl;
            mImpl = nullptr;
            return false;
        }
        return true;
    }

    bool Window::ShouldClose() const {
        return mImpl != nullptr && glfwWindowShouldClose(mImpl->mWindow);
    }

    void Window::PollEvents() {
        if (mImpl != nullptr) {
            glfwPollEvents();
        }
    }

    uintptr_t Window::GetSurfaceHandle() const {
        return reinterpret_cast<uintptr_t>(mImpl->mSurface);
    }

    uintptr_t Window::GetHandle() const {
        return mImpl != nullptr ? reinterpret_cast<uintptr_t>(mImpl->mWindow) : 0;
    }
}// namespace moe::neo