#include <cstdio>
#include <cstdlib>

#include <volk.h>
#include <VkBootstrap.h>

// Headless Vulkan bootstrap smoke test.
// Verifies the Vulkan toolchain works on this machine without needing a
// window or assets: create an instance, pick a physical device with the same
// feature set the engine requires, create a logical device, then tear down.
int main() {
    if (volkInitialize() != VK_SUCCESS) {
        std::fprintf(stderr, "volkInitialize failed\n");
        return EXIT_FAILURE;
    }

    vkb::InstanceBuilder instanceBuilder;
    auto instanceResult = instanceBuilder.set_app_name("moe-vk-smoke")
            .set_app_version(0, 1, 0)
            .enable_extension(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME)
            .require_api_version(1, 3, 0)
            .build();
    if (!instanceResult) {
        std::fprintf(stderr, "Failed to create Vulkan instance: %s\n", instanceResult.error().message().c_str());
        return EXIT_FAILURE;
    }
    auto vkbInstance = *instanceResult;
    volkLoadInstance(vkbInstance.instance);

    VkSurfaceKHR headlessSurface = VK_NULL_HANDLE;
    {
        VkHeadlessSurfaceCreateInfoEXT surfaceInfo = {
                .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
        };
        if (vkCreateHeadlessSurfaceEXT(vkbInstance.instance, &surfaceInfo, nullptr, &headlessSurface) != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to create headless surface\n");
            vkb::destroy_instance(vkbInstance);
            return EXIT_FAILURE;
        }
    }

    VkPhysicalDeviceFeatures requiredFeatures = {
            .imageCubeArray = VK_TRUE,
            .geometryShader = VK_TRUE,
            .depthClamp = VK_TRUE,
            .samplerAnisotropy = VK_TRUE,
            .shaderStorageImageMultisample = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features features13 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features features12 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .descriptorIndexing = VK_TRUE,
            .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
            .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
            .descriptorBindingPartiallyBound = VK_TRUE,
            .descriptorBindingVariableDescriptorCount = VK_TRUE,
            .runtimeDescriptorArray = VK_TRUE,
            .scalarBlockLayout = VK_TRUE,
            .bufferDeviceAddress = VK_TRUE,
    };

    vkb::PhysicalDeviceSelector selector{vkbInstance};
    auto selectorResult = selector.set_minimum_version(1, 3)
            .set_required_features(requiredFeatures)
            .set_required_features_12(features12)
            .set_required_features_13(features13)
            .add_required_extension("VK_EXT_descriptor_indexing")
            .add_required_extension("VK_KHR_shader_non_semantic_info")
            .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
            .allow_any_gpu_device_type(true)
            .set_surface(headlessSurface)
            .select();
    if (!selectorResult) {
        std::fprintf(stderr, "Failed to select a physical device: %s\n", selectorResult.error().message().c_str());
        vkb::destroy_instance(vkbInstance);
        return EXIT_FAILURE;
    }
    auto vkbPhysicalDevice = *selectorResult;
    std::printf("Selected GPU: %s\n", vkbPhysicalDevice.properties.deviceName);

    vkb::DeviceBuilder deviceBuilder{vkbPhysicalDevice};
    auto deviceResult = deviceBuilder.build();
    if (!deviceResult) {
        std::fprintf(stderr, "Failed to create logical device: %s\n", deviceResult.error().message().c_str());
        vkb::destroy_instance(vkbInstance);
        return EXIT_FAILURE;
    }
    auto vkbDevice = *deviceResult;
    volkLoadDevice(vkbDevice.device);

    vkb::destroy_device(vkbDevice);
    vkDestroySurfaceKHR(vkbInstance.instance, headlessSurface, nullptr);
    vkb::destroy_instance(vkbInstance);
    std::printf("Vulkan smoke test passed.\n");
    return EXIT_SUCCESS;
}