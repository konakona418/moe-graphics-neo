#include "UI/DebugUI.hpp"

#include <Core/Error.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/Device.hpp>
#include <RHI/Swapchain.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <volk.h>

#include <cstdio>

namespace moe::ui {
    namespace {
        // The swapchain only ever uses these four 8-bit RGBA formats; anything
        // else means the RHI added a new one and this mapping went stale.
        VkFormat ToVkSwapchainFormat(moe::rhi::Format format) {
            switch (format) {
                case moe::rhi::Format::kR8G8B8A8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
                case moe::rhi::Format::kR8G8B8A8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
                case moe::rhi::Format::kB8G8R8A8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
                case moe::rhi::Format::kB8G8R8A8Srgb: return VK_FORMAT_B8G8R8A8_SRGB;
                default: return VK_FORMAT_UNDEFINED;
            }
        }

        // The ImGui pipeline must match the render pass's sample count.
        VkSampleCountFlagBits ToVkSampleCount(uint32_t samples) {
            switch (samples) {
                case 2: return VK_SAMPLE_COUNT_2_BIT;
                case 4: return VK_SAMPLE_COUNT_4_BIT;
                case 8: return VK_SAMPLE_COUNT_8_BIT;
                default: return VK_SAMPLE_COUNT_1_BIT;
            }
        }
    }// namespace

    struct DebugUIImpl {
        ImGuiContext* mContext{nullptr};
        VkDevice mDevice{VK_NULL_HANDLE};
        VkDescriptorPool mPool{VK_NULL_HANDLE};
        bool mBackendsReady{false};
    };

    DebugUI::DebugUI() = default;

    DebugUI::~DebugUI() {
        if (mImpl != nullptr) {
            std::fprintf(stderr, "[ui] DebugUI leaked: Destroy() not called\n");
            std::abort();
        }
    }

    void DebugUI::Destroy() {
        if (mImpl == nullptr) {
            return;
        }
        if (mImpl->mBackendsReady) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext(mImpl->mContext);
        }
        if (mImpl->mPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(mImpl->mDevice, mImpl->mPool, nullptr);
        }
        mImpl.reset();
    }

    bool DebugUI::Init(moe::rhi::Device& device, moe::rhi::Swapchain& swapchain,
            uintptr_t glfwWindow) {
        mImpl = std::make_unique<DebugUIImpl>();

        moe::rhi::RhiVulkanHandles handles;
        if (!device.GetVulkanHandles(handles)) {
            mImpl.reset();
            return moe::Fail("GetVulkanHandles failed");
        }
        mImpl->mDevice = reinterpret_cast<VkDevice>(handles.mDevice);

        const VkFormat colorFormat = ToVkSwapchainFormat(swapchain.GetFormat());
        if (colorFormat == VK_FORMAT_UNDEFINED) {
            mImpl.reset();
            return moe::Fail("unsupported swapchain format for ImGui");
        }

        // Descriptor pool for ImGui (same spec as the old engine's initImGUI).
        const VkDescriptorPoolSize poolSizes[] = {
                {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000},
        };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1000;
        poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
        poolInfo.pPoolSizes = poolSizes;
        if (vkCreateDescriptorPool(mImpl->mDevice, &poolInfo, nullptr, &mImpl->mPool) != VK_SUCCESS) {
            mImpl.reset();
            return moe::Fail("vkCreateDescriptorPool failed");
        }

        IMGUI_CHECKVERSION();
        mImpl->mContext = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();

        if (!ImGui_ImplGlfw_InitForVulkan(reinterpret_cast<GLFWwindow*>(glfwWindow), true)) {
            ImGui::DestroyContext(mImpl->mContext);
            mImpl.reset();
            return moe::Fail("ImGui_ImplGlfw_InitForVulkan failed");
        }

        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.Instance = reinterpret_cast<VkInstance>(handles.mInstance);
        initInfo.PhysicalDevice = reinterpret_cast<VkPhysicalDevice>(handles.mPhysicalDevice);
        initInfo.Device = mImpl->mDevice;
        initInfo.QueueFamily = handles.mGraphicsQueueFamily;
        initInfo.Queue = reinterpret_cast<VkQueue>(handles.mGraphicsQueue);
        initInfo.DescriptorPool = mImpl->mPool;
        initInfo.MinImageCount = 2;
        initInfo.ImageCount = 2;
        initInfo.MSAASamples = ToVkSampleCount(swapchain.GetSampleCount());
        initInfo.UseDynamicRendering = true;
        initInfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
        if (!ImGui_ImplVulkan_Init(&initInfo)) {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext(mImpl->mContext);
            vkDestroyDescriptorPool(mImpl->mDevice, mImpl->mPool, nullptr);
            mImpl.reset();
            return moe::Fail("ImGui_ImplVulkan_Init failed");
        }
        mImpl->mBackendsReady = true;
        return true;
    }

    void DebugUI::BeginFrame(float deltaSeconds) {
        ImGuiIO& io = ImGui::GetIO();
        io.DeltaTime = deltaSeconds > 0.0f ? deltaSeconds : 1.0f / 60.0f;
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void DebugUI::Render(moe::rhi::CommandList& cmd) {
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                reinterpret_cast<VkCommandBuffer>(cmd.GetVulkanHandle()));
    }
}// namespace moe::ui
