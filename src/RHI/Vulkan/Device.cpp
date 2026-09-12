#include "RHI/Device.hpp"

#include "RHI/Buffer.hpp"
#include "RHI/CommandList.hpp"
#include "RHI/DescriptorSet.hpp"
#include "RHI/Fence.hpp"
#include "RHI/Image.hpp"
#include "RHI/Pipeline.hpp"
#include "RHI/PipelineCache.hpp"
#include "RHI/Queue.hpp"
#include "RHI/Sampler.hpp"
#include "RHI/Swapchain.hpp"
#include "RHI/TimelineSemaphore.hpp"
#include "Core/Defer.hpp"
#include "Core/Error.hpp"
#include "Core/Logger.hpp"
#include <Core/Profile.hpp>
#include "Mappings.hpp"
#include "RhiAssert.hpp"
#include "RhiInternal.hpp"

#include <VkBootstrap.h>

#include <algorithm>
#include <string>
#include <utility>

namespace moe::rhi {
    namespace {
        VkBufferUsageFlags ToVulkanBufferUsage(BufferUsage usage) {
            VkBufferUsageFlags flags = 0;
            if (HasFlag(usage, BufferUsage::kUniform)) {
                flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            }
            if (HasFlag(usage, BufferUsage::kStorage)) {
                flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            }
            if (HasFlag(usage, BufferUsage::kVertex)) {
                flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            }
            if (HasFlag(usage, BufferUsage::kIndex)) {
                flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            }
            if (HasFlag(usage, BufferUsage::kTransferSrc)) {
                flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            }
            if (HasFlag(usage, BufferUsage::kTransferDst)) {
                flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            }
            return flags;
        }

        void Teardown(DeviceImpl* impl) {
            if (impl == nullptr) {
                return;
            }
            impl->FlushDeferredDeletions();
            if (impl->mAllocator != VK_NULL_HANDLE) {
                vmaDestroyAllocator(impl->mAllocator);
            }
            if (impl->mDevice != VK_NULL_HANDLE) {
                for (const VkCommandPool pool : impl->mCommandPools) {
                    if (pool != VK_NULL_HANDLE) {
                        vkDestroyCommandPool(impl->mDevice, pool, nullptr);
                    }
                }
                vkDestroyDevice(impl->mDevice, nullptr);
            }
            if (impl->mInstance.instance != VK_NULL_HANDLE) {
                vkb::destroy_instance(impl->mInstance);
            }
        }
    }// namespace

    Device::Device() = default;

    bool Device::Create(const DeviceCreateInfo& info, Device& outDevice) {
        MOE_PROFILE_ZONE();
        if (info.mPipelineCache == nullptr) {
            return Fail("Device requires an injected PipelineCache (null cache is a fatal error)");
        }

        outDevice.mImpl = std::make_unique<DeviceImpl>();
        auto* impl = outDevice.mImpl.get();
        bool success = false;
        Defer rollback{[&] {
            if (!success) {
                Teardown(outDevice.mImpl.get());
                outDevice.mImpl.reset();
            }
        }};

        if (volkInitialize() != VK_SUCCESS) {
            return Fail("volkInitialize failed");
        }

        vkb::InstanceBuilder instanceBuilder;
        instanceBuilder.set_app_name(info.mApplicationName.c_str())
                .set_app_version(0, 1, 0)
                .request_validation_layers(info.mEnableValidation)
                .require_api_version(1, 3, 0);
        if (info.mEnablePresent) {
            instanceBuilder.enable_extension(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(_WIN32)
            instanceBuilder.enable_extension("VK_KHR_win32_surface");
#else
            // Linux: request the surface extensions that GLFW may pick between
            instanceBuilder.enable_extension("VK_KHR_xlib_surface")
                    .enable_extension("VK_KHR_xcb_surface")
                    .enable_extension("VK_KHR_wayland_surface");
#endif
        }
        auto instanceResult = instanceBuilder.build();
        if (!instanceResult) {
            return Fail("Failed to create Vulkan instance: " + instanceResult.error().message());
        }
        auto vkbInstance = *instanceResult;
        impl->mInstance = std::move(vkbInstance);
        volkLoadInstance(impl->mInstance.instance);

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
                .scalarBlockLayout = VK_TRUE,
                .timelineSemaphore = VK_TRUE,
                .bufferDeviceAddress = VK_TRUE,
        };
        if (info.mEnableDescriptorIndexing) {
            // bindless descriptor arrays (mirrors the old engine's request list)
            features12.descriptorIndexing = VK_TRUE;
            features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
            features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
            features12.descriptorBindingPartiallyBound = VK_TRUE;
            features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
            features12.runtimeDescriptorArray = VK_TRUE;
        }

        vkb::PhysicalDeviceSelector selector{impl->mInstance};
        selector.set_minimum_version(1, 3)
                .set_required_features(requiredFeatures)
                .set_required_features_12(features12)
                .set_required_features_13(features13)
                .require_present(false)
                .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                .allow_any_gpu_device_type(true);
        if (info.mEnablePresent) {
            selector.add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }
        auto selectorResult = selector.select();
        if (!selectorResult) {
            return Fail("Failed to select a physical device: " + selectorResult.error().message());
        }
        auto vkbPhysicalDevice = *selectorResult;
        impl->mPhysicalDevice = vkbPhysicalDevice.physical_device;
        // highest color+depth sample count supported for framebuffer attachments
        {
            const VkSampleCountFlags common = vkbPhysicalDevice.properties.limits.framebufferColorSampleCounts
                    & vkbPhysicalDevice.properties.limits.framebufferDepthSampleCounts;
            for (const uint32_t samples : {8u, 4u, 2u}) {
                if ((common & ToVkSampleCount(samples)) != 0) {
                    impl->mMaxSampleCount = samples;
                    break;
                }
            }
        }
        moe::Logger::Info("RHI selected GPU: {}", vkbPhysicalDevice.properties.deviceName);

        vkb::DeviceBuilder deviceBuilder{vkbPhysicalDevice};
        auto deviceResult = deviceBuilder.build();
        if (!deviceResult) {
            return Fail("Failed to create logical device: " + deviceResult.error().message());
        }
        auto vkbDevice = *deviceResult;
        impl->mDevice = vkbDevice.device;
        volkLoadDevice(impl->mDevice);

        auto queueResult = vkbDevice.get_queue(vkb::QueueType::graphics);
        if (!queueResult) {
            return Fail("Failed to get graphics queue: " + queueResult.error().message());
        }
        impl->mGraphicsQueue = *queueResult;
        auto queueIndexResult = vkbDevice.get_queue_index(vkb::QueueType::graphics);
        impl->mGraphicsQueueFamily = queueIndexResult.has_value() ? *queueIndexResult : 0;

        // A compute/transfer queue separate from graphics when the device has
        // one (vk-bootstrap creates a queue per family); otherwise fall back to
        // the graphics queue so single-family devices still work.
        if (auto computeQueue = vkbDevice.get_queue(vkb::QueueType::compute)) {
            impl->mComputeQueue = *computeQueue;
        } else {
            impl->mComputeQueue = impl->mGraphicsQueue;
        }
        if (auto computeIndex = vkbDevice.get_queue_index(vkb::QueueType::compute)) {
            impl->mComputeQueueFamily = *computeIndex;
        } else {
            impl->mComputeQueueFamily = impl->mGraphicsQueueFamily;
        }
        // Create the graphics pool now (its queueFamilyIndex must be the real
        // family, never a hardcoded 0).
        if (impl->GetOrCreateCommandPool(impl->mGraphicsQueueFamily) == VK_NULL_HANDLE) {
            return Fail("Failed to create command pool");
        }

        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorInfo.instance = impl->mInstance.instance;
        allocatorInfo.physicalDevice = impl->mPhysicalDevice;
        allocatorInfo.device = impl->mDevice;
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

        VmaVulkanFunctions vulkanFunctions{};
        vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        allocatorInfo.pVulkanFunctions = &vulkanFunctions;

        if (vmaCreateAllocator(&allocatorInfo, &impl->mAllocator) != VK_SUCCESS) {
            return Fail("Failed to create memory allocator");
        }

        if (!info.mPipelineCache->Create(outDevice)) {
            return Fail("PipelineCache::Create failed");
        }

        outDevice.mPipelineCache = info.mPipelineCache;
        success = true;
        return true;
    }

    Device::~Device() {
        MOE_RHI_ASSERT(mImpl == nullptr, "Device leaked: Destroy() not called");
    }

    void Device::Destroy() {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr) {
            return;
        }
        Teardown(mImpl.get());
        mImpl.reset();
        mPipelineCache = nullptr;
    }

    bool Device::CreateBuffer(const BufferCreateInfo& info, Buffer& outBuffer) {
        MOE_PROFILE_ZONE();
        outBuffer.mImpl = std::make_unique<BufferImpl>();
        auto* impl = outBuffer.mImpl.get();
        impl->mDevice = mImpl.get();
        impl->mSize = info.mSize;
        impl->mCpuVisible = info.mCpuVisible;
        impl->mUsage = info.mUsage;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = info.mSize;
        bufferInfo.usage = ToVulkanBufferUsage(info.mUsage) | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        uint32_t sharingFamilies[2] = {mImpl->mGraphicsQueueFamily, mImpl->mComputeQueueFamily};
        const uint32_t sharingFamilyCount = info.mSharedAcrossQueues
                        && mImpl->mComputeQueueFamily != mImpl->mGraphicsQueueFamily
                ? 2u
                : 1u;
        if (sharingFamilyCount > 1) {
            bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bufferInfo.queueFamilyIndexCount = sharingFamilyCount;
            bufferInfo.pQueueFamilyIndices = sharingFamilies;
        } else {
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VmaAllocationCreateInfo allocInfo{};
        if (info.mCpuVisible) {
            allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        } else {
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        }

        const VkResult result = vmaCreateBuffer(mImpl->mAllocator, &bufferInfo, &allocInfo,
                &impl->mBuffer, &impl->mAllocation, nullptr);
        if (result != VK_SUCCESS) {
            outBuffer.mImpl.reset();
            return Fail("Failed to allocate buffer (" + std::to_string(info.mSize)
                    + " bytes, VkResult " + std::to_string(static_cast<int>(result)) + ")");
        }

        VkBufferDeviceAddressInfo addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addressInfo.buffer = impl->mBuffer;
        impl->mDeviceAddress = vkGetBufferDeviceAddress(mImpl->mDevice, &addressInfo);
        return true;
    }

    bool Device::CreateImage(const ImageCreateInfo& info, Image& outImage) {
        MOE_PROFILE_ZONE();
        outImage.mImpl = std::make_unique<ImageImpl>();
        auto* impl = outImage.mImpl.get();
        impl->mDevice = mImpl.get();
        impl->mType = info.mType;
        impl->mWidth = info.mWidth;
        impl->mHeight = info.mHeight;
        impl->mDepth = info.mDepth;
        impl->mMipLevels = info.mMipLevels;
        impl->mLayerCount = info.mLayerCount;
        impl->mFormat = info.mFormat;
        impl->mUsage = info.mUsage;

        const uint32_t layerCount = info.mType == ImageType::kCube ? 6 : info.mLayerCount;
        const VkImageCreateFlags flags = info.mType == ImageType::kCube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = ToVkImageType(info.mType);
        imageInfo.format = ToVkFormat(info.mFormat);
        imageInfo.extent = {info.mWidth, info.mHeight, info.mDepth};
        imageInfo.mipLevels = info.mMipLevels;
        imageInfo.arrayLayers = layerCount;
        imageInfo.samples = ToVkSampleCount(info.mSampleCount);
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = ToVkImageUsage(info.mUsage);
        uint32_t sharingFamilies[2] = {mImpl->mGraphicsQueueFamily, mImpl->mComputeQueueFamily};
        const uint32_t sharingFamilyCount = info.mSharedAcrossQueues
                        && mImpl->mComputeQueueFamily != mImpl->mGraphicsQueueFamily
                ? 2u
                : 1u;
        if (sharingFamilyCount > 1) {
            imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
            imageInfo.queueFamilyIndexCount = sharingFamilyCount;
            imageInfo.pQueueFamilyIndices = sharingFamilies;
        } else {
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.flags = flags;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        if (vmaCreateImage(mImpl->mAllocator, &imageInfo, &allocInfo,
                    &impl->mImage, &impl->mAllocation, nullptr)
                != VK_SUCCESS) {
            outImage.mImpl.reset();
            return Fail("Failed to allocate image");
        }

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = impl->mImage;
        viewInfo.viewType = info.mType == ImageType::k3D ? VK_IMAGE_VIEW_TYPE_3D
                : info.mType == ImageType::kCube ? VK_IMAGE_VIEW_TYPE_CUBE
                : VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = ToVkFormat(info.mFormat);
        viewInfo.subresourceRange.aspectMask = ToVkImageAspect(info.mFormat);
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = info.mMipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = layerCount;

        if (vkCreateImageView(mImpl->mDevice, &viewInfo, nullptr, &impl->mView) != VK_SUCCESS) {
            vmaDestroyImage(mImpl->mAllocator, impl->mImage, impl->mAllocation);
            outImage.mImpl.reset();
            return Fail("Failed to create image view");
        }
        return true;
    }

    bool Device::CreateSampler(const SamplerCreateInfo& info, Sampler& outSampler) {
        MOE_PROFILE_ZONE();
        outSampler.mImpl = std::make_unique<SamplerImpl>();
        auto* impl = outSampler.mImpl.get();
        impl->mDevice = mImpl.get();

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = ToVkFilter(info.mMagFilter);
        samplerInfo.minFilter = ToVkFilter(info.mMinFilter);
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = ToVkAddressMode(info.mAddressModeU);
        samplerInfo.addressModeV = ToVkAddressMode(info.mAddressModeV);
        samplerInfo.addressModeW = ToVkAddressMode(info.mAddressModeW);
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

        if (vkCreateSampler(mImpl->mDevice, &samplerInfo, nullptr, &impl->mSampler) != VK_SUCCESS) {
            outSampler.mImpl.reset();
            return Fail("Failed to create sampler");
        }
        return true;
    }

    bool Device::CreateTimelineSemaphore(TimelineSemaphore& outSemaphore, uint64_t initialValue) {
        MOE_PROFILE_ZONE();
        outSemaphore.mImpl = std::make_unique<TimelineSemaphoreImpl>();
        auto* impl = outSemaphore.mImpl.get();
        impl->mDevice = mImpl.get();

        VkSemaphoreTypeCreateInfo typeInfo{};
        typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        typeInfo.initialValue = initialValue;

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreInfo.pNext = &typeInfo;
        if (vkCreateSemaphore(mImpl->mDevice, &semaphoreInfo, nullptr, &impl->mSemaphore)
                != VK_SUCCESS) {
            outSemaphore.mImpl.reset();
            return Fail("Failed to create timeline semaphore");
        }
        return true;
    }

    bool Device::CreateFence(Fence& outFence, bool signaled) {
        MOE_PROFILE_ZONE();
        outFence.mImpl = std::make_unique<FenceImpl>();
        auto* impl = outFence.mImpl.get();
        impl->mDevice = mImpl.get();

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;
        if (vkCreateFence(mImpl->mDevice, &fenceInfo, nullptr, &impl->mFence) != VK_SUCCESS) {
            outFence.mImpl.reset();
            return Fail("Failed to create fence");
        }
        return true;
    }

    bool Device::CreateCommandList(QueueType type, CommandList& outCommandList) {
        MOE_PROFILE_ZONE();
        const uint32_t family = type == QueueType::kGraphics
                ? mImpl->mGraphicsQueueFamily
                : mImpl->mComputeQueueFamily;
        const VkCommandPool pool = mImpl->GetOrCreateCommandPool(family);
        if (pool == VK_NULL_HANDLE) {
            return Fail("Failed to create command pool for queue family "
                    + std::to_string(family));
        }
        outCommandList.mImpl = std::make_unique<CommandListImpl>();
        auto* impl = outCommandList.mImpl.get();
        impl->mDevice = mImpl.get();
        impl->mPool = pool;

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(mImpl->mDevice, &allocInfo, &impl->mCommandBuffer) != VK_SUCCESS) {
            outCommandList.mImpl.reset();
            return Fail("Failed to allocate command buffer");
        }
        return true;
    }

    bool Device::CreateCommandList(CommandList& outCommandList) {
        return CreateCommandList(QueueType::kGraphics, outCommandList);
    }

    bool Device::CreateSwapchain(uintptr_t surfaceHandle, uint32_t width, uint32_t height,
            Swapchain& outSwapchain, uint32_t sampleCount) {
        MOE_PROFILE_ZONE();
        outSwapchain.mImpl = std::make_unique<SwapchainImpl>();
        auto* impl = outSwapchain.mImpl.get();
        impl->mDevice = mImpl.get();
        impl->mSurface = reinterpret_cast<VkSurfaceKHR>(surfaceHandle);
        impl->mWidth = width;
        impl->mHeight = height;
        bool success = false;
        Defer rollback{[&] {
            if (!success) {
                if (outSwapchain.mImpl != nullptr) {
                    outSwapchain.mImpl->mMsaaImage.Destroy();
                }
                outSwapchain.mImpl.reset();
            }
        }};

        VkBool32 presentSupported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(mImpl->mPhysicalDevice, mImpl->mGraphicsQueueFamily,
                impl->mSurface, &presentSupported);
        if (presentSupported != VK_TRUE) {
            return Fail("Surface does not support present on the graphics queue");
        }

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(mImpl->mPhysicalDevice, impl->mSurface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(mImpl->mPhysicalDevice, impl->mSurface, &formatCount, formats.data());
        if (formats.empty()) {
            return Fail("No surface formats available");
        }
        VkSurfaceFormatKHR chosen = formats[0];
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_R8G8B8A8_SRGB) {
                chosen = format;
                break;
            }
        }
        // Map the chosen Vk format back to a backend-agnostic one (B8G8R8A8 is
        // a common swapchain order; mismapping it would swap the R/B channels).
        switch (chosen.format) {
            case VK_FORMAT_R8G8B8A8_SRGB: impl->mFormat = Format::kR8G8B8A8Srgb; break;
            case VK_FORMAT_R8G8B8A8_UNORM: impl->mFormat = Format::kR8G8B8A8Unorm; break;
            case VK_FORMAT_B8G8R8A8_SRGB: impl->mFormat = Format::kB8G8R8A8Srgb; break;
            case VK_FORMAT_B8G8R8A8_UNORM: impl->mFormat = Format::kB8G8R8A8Unorm; break;
            default: impl->mFormat = Format::kR8G8B8A8Unorm; break;
        }

        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mImpl->mPhysicalDevice, impl->mSurface, &caps);

        uint32_t imageCount = std::max(2u, caps.minImageCount);
        if (caps.maxImageCount > 0) {
            imageCount = std::min(imageCount, caps.maxImageCount);
        }

        VkExtent2D extent{width, height};
        if (caps.currentExtent.width != 0xFFFFFFFF) {
            extent = caps.currentExtent;
        }
        impl->mWidth = extent.width;
        impl->mHeight = extent.height;

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = impl->mSurface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = chosen.format;
        createInfo.imageColorSpace = chosen.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = caps.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        createInfo.clipped = VK_TRUE;
        if (vkCreateSwapchainKHR(mImpl->mDevice, &createInfo, nullptr, &impl->mSwapchain) != VK_SUCCESS) {
            return Fail("Failed to create swapchain");
        }

        uint32_t swapImageCount = 0;
        vkGetSwapchainImagesKHR(mImpl->mDevice, impl->mSwapchain, &swapImageCount, nullptr);
        impl->mImages.resize(swapImageCount);
        vkGetSwapchainImagesKHR(mImpl->mDevice, impl->mSwapchain, &swapImageCount, impl->mImages.data());

        impl->mImageViews.resize(swapImageCount);
        for (uint32_t i = 0; i < swapImageCount; ++i) {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = impl->mImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = chosen.format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(mImpl->mDevice, &viewInfo, nullptr, &impl->mImageViews[i]) != VK_SUCCESS) {
                return Fail("Failed to create swapchain image view");
            }
        }

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // first AcquireImage must not block
        if (vkCreateSemaphore(mImpl->mDevice, &semaphoreInfo, nullptr, &impl->mImageAvailable) != VK_SUCCESS
                || vkCreateFence(mImpl->mDevice, &fenceInfo, nullptr, &impl->mInFlight) != VK_SUCCESS) {
            return Fail("Failed to create swapchain sync objects");
        }
        impl->mRenderFinished.resize(swapImageCount);
        for (auto& semaphore : impl->mRenderFinished) {
            if (vkCreateSemaphore(mImpl->mDevice, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
                return Fail("Failed to create swapchain render-finished semaphores");
            }
        }

        if (sampleCount > 1) {
            if (sampleCount > mImpl->mMaxSampleCount) {
                return Fail("Swapchain sample count exceeds device support");
            }
            ImageCreateInfo msaaInfo{};
            msaaInfo.mType = ImageType::k2D;
            msaaInfo.mWidth = impl->mWidth;
            msaaInfo.mHeight = impl->mHeight;
            msaaInfo.mFormat = impl->mFormat;
            msaaInfo.mUsage = ImageUsage::kColorAttachment;
            msaaInfo.mSampleCount = sampleCount;
            if (!CreateImage(msaaInfo, impl->mMsaaImage)) {
                return Fail("Failed to create multisampled swapchain image: " + moe::Error::Get());
            }
            impl->mSampleCount = sampleCount;
            moe::Logger::Info("Swapchain: {}x MSAA", sampleCount);
        }
        success = true;
        return true;
    }

    bool Device::CreateDescriptorSet(const DescriptorSetLayout& layout, DescriptorSet& outSet) {
        MOE_PROFILE_ZONE();
        if (!layout.mImpl || layout.mImpl->mSetLayout == VK_NULL_HANDLE) {
            return Fail("Invalid descriptor set layout");
        }

        std::vector<VkDescriptorPoolSize> poolSizes;
        for (const auto& binding : layout.mImpl->mBindings) {
            bool found = false;
            for (auto& size : poolSizes) {
                if (size.type == ToVkDescriptorType(binding.mType)) {
                    size.descriptorCount += binding.mCount;
                    found = true;
                    break;
                }
            }
            if (!found) {
                poolSizes.push_back({ToVkDescriptorType(binding.mType), binding.mCount});
            }
        }

        outSet.mImpl = std::make_unique<DescriptorSetImpl>();
        outSet.mImpl->mDevice = mImpl.get();

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(mImpl->mDevice, &poolInfo, nullptr, &outSet.mImpl->mPool) != VK_SUCCESS) {
            outSet.mImpl.reset();
            return Fail("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo setInfo{};
        setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setInfo.descriptorPool = outSet.mImpl->mPool;
        setInfo.descriptorSetCount = 1;
        setInfo.pSetLayouts = &layout.mImpl->mSetLayout;
        if (vkAllocateDescriptorSets(mImpl->mDevice, &setInfo, &outSet.mImpl->mSet) != VK_SUCCESS) {
            outSet.mImpl.reset();
            return Fail("Failed to allocate descriptor set");
        }
        return true;
    }

    bool Device::GetOrCreateGraphicsPipeline(const GraphicsPipelineState& state, GraphicsPipeline& out) {
        MOE_PROFILE_ZONE();
        if (!mPipelineCache->GetOrCreateGraphics(state, out)) {
            return Fail("PipelineCache::GetOrCreateGraphics failed");
        }
        return true;
    }

    bool Device::GetOrCreateComputePipeline(const ComputePipelineState& state, ComputePipeline& out) {
        MOE_PROFILE_ZONE();
        if (!mPipelineCache->GetOrCreateCompute(state, out)) {
            return Fail("PipelineCache::GetOrCreateCompute failed");
        }
        return true;
    }

    bool Device::Submit(const CommandList& commandList, bool waitForCompletion) {
        MOE_PROFILE_ZONE();
        Queue queue;
        if (!GetQueue(QueueType::kGraphics, queue)) {
            return Fail("Failed to get the graphics queue");
        }
        if (!queue.Submit(commandList, waitForCompletion)) {
            return Fail("Failed to submit command buffer");
        }
        return true;
    }

    bool Device::Submit(const CommandList& commandList, const SubmitInfo& submitInfo, Fence* fence) {
        MOE_PROFILE_ZONE();
        Queue queue;
        if (!GetQueue(QueueType::kGraphics, queue)) {
            return Fail("Failed to get the graphics queue");
        }
        if (!queue.Submit(commandList, submitInfo, fence)) {
            return Fail("Failed to submit command buffer");
        }
        return true;
    }

    bool Device::WaitIdle() {
        MOE_PROFILE_ZONE();
        if (vkDeviceWaitIdle(mImpl->mDevice) != VK_SUCCESS) {
            return Fail("Failed to wait for device idle");
        }
        mImpl->FlushDeferredDeletions();
        return true;
    }

    bool Device::GetQueue(QueueType type, Queue& outQueue) const {
        outQueue.mDevice = reinterpret_cast<uintptr_t>(mImpl->mDevice);
        outQueue.mType = type;
        switch (type) {
            case QueueType::kGraphics:
                outQueue.mQueue = reinterpret_cast<uintptr_t>(mImpl->mGraphicsQueue);
                outQueue.mFamily = mImpl->mGraphicsQueueFamily;
                break;
            case QueueType::kCompute:
            case QueueType::kTransfer:
                outQueue.mQueue = reinterpret_cast<uintptr_t>(mImpl->mComputeQueue);
                outQueue.mFamily = mImpl->mComputeQueueFamily;
                break;
        }
        return outQueue.mQueue != 0;
    }

    bool Device::GetInstanceHandle(uintptr_t& outInstance) const {
        if (mImpl == nullptr || mImpl->mInstance.instance == VK_NULL_HANDLE) {
            return false;
        }
        outInstance = reinterpret_cast<uintptr_t>(mImpl->mInstance.instance);
        return true;
    }

    uint32_t Device::GetMaxSampleCount() const {
        return mImpl != nullptr ? mImpl->mMaxSampleCount : 1;
    }

    Format Device::GetDepthStencilFormat() const {
        if (mImpl == nullptr) {
            return Format::kD32FloatS8Uint;
        }
        if (mImpl->mDepthStencilFormat != Format::kUndefined) {
            return mImpl->mDepthStencilFormat;
        }
        // Vulkan guarantees at least one combined depth-stencil format; prefer
        // the smaller D24_UNORM_S8_UINT, fall back to D32_SFLOAT_S8_UINT.
        const Format candidates[] = {Format::kD24UnormS8Uint, Format::kD32FloatS8Uint};
        for (const Format candidate : candidates) {
            VkPhysicalDeviceImageFormatInfo2 formatInfo{};
            formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
            formatInfo.format = ToVkFormat(candidate);
            formatInfo.type = VK_IMAGE_TYPE_2D;
            formatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            formatInfo.usage =
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            VkImageFormatProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
            if (vkGetPhysicalDeviceImageFormatProperties2(mImpl->mPhysicalDevice, &formatInfo,
                        &properties) == VK_SUCCESS) {
                mImpl->mDepthStencilFormat = candidate;
                return candidate;
            }
        }
        mImpl->mDepthStencilFormat = Format::kD32FloatS8Uint;
        return mImpl->mDepthStencilFormat;
    }

    bool Device::GetVulkanHandles(RhiVulkanHandles& outHandles) const {
        if (mImpl == nullptr || mImpl->mDevice == VK_NULL_HANDLE) {
            return false;
        }
        outHandles.mInstance = reinterpret_cast<uintptr_t>(mImpl->mInstance.instance);
        outHandles.mPhysicalDevice = reinterpret_cast<uintptr_t>(mImpl->mPhysicalDevice);
        outHandles.mDevice = reinterpret_cast<uintptr_t>(mImpl->mDevice);
        outHandles.mGraphicsQueue = reinterpret_cast<uintptr_t>(mImpl->mGraphicsQueue);
        outHandles.mGraphicsQueueFamily = mImpl->mGraphicsQueueFamily;
        return true;
    }
}// namespace moe::rhi