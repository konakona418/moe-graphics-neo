#include "Neo/Window.hpp"
#include <Core/Profile.hpp>

#include <Core/Error.hpp>

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
        MOE_PROFILE_ZONE();
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
            const char* title) {
        MOE_PROFILE_ZONE();
        mImpl = new Impl();

        if (glfwInit() != GLFW_TRUE) {
            delete mImpl;
            mImpl = nullptr;
            return moe::Fail("glfwInit failed");
        }
        if (glfwVulkanSupported() != GLFW_TRUE) {
            delete mImpl;
            mImpl = nullptr;
            return moe::Fail("glfw: Vulkan not supported");
        }
        // tell GLFW to resolve instance functions through the same loader volk uses
        glfwInitVulkanLoader(vkGetInstanceProcAddr);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        mImpl->mWindow = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (mImpl->mWindow == nullptr) {
            delete mImpl;
            mImpl = nullptr;
            return moe::Fail("glfwCreateWindow failed");
        }

        uintptr_t instanceHandle = 0;
        if (!device.GetInstanceHandle(instanceHandle)) {
            glfwDestroyWindow(mImpl->mWindow);
            delete mImpl;
            mImpl = nullptr;
            return moe::Fail("device has no instance (mEnablePresent required)");
        }
        mImpl->mInstance = reinterpret_cast<VkInstance>(instanceHandle);

        if (glfwCreateWindowSurface(mImpl->mInstance, mImpl->mWindow, nullptr, &mImpl->mSurface) != VK_SUCCESS) {
            glfwDestroyWindow(mImpl->mWindow);
            delete mImpl;
            mImpl = nullptr;
            return moe::Fail("glfwCreateWindowSurface failed");
        }
        return true;
    }

    bool Window::ShouldClose() const {
        return mImpl != nullptr && glfwWindowShouldClose(mImpl->mWindow);
    }

    void Window::PollEvents() {
        MOE_PROFILE_ZONE();
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