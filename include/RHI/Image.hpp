#pragma once

#include "RHI/RHICommon.hpp"

#include <memory>

namespace moe::rhi {
    class Device;
    class CommandList;
    class DescriptorSet;
    struct ImageImpl;

    // GPU image (2D/3D/cube). Created via Device::CreateImage; destroyed
    // explicitly via Destroy() (the destructor is a leak trap). The image has
    // no layout state of its own — layouts are explicit ImageBarrier
    // parameters (the RenderGraph tracks them for the passes it orchestrates).
    class Image {
    public:
        Image();
        ~Image();

        Image(const Image&) = delete;
        Image& operator=(const Image&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the image
        // was created but not destroyed (leak trap).
        void Destroy();

        ImageType GetType() const;
        uint32_t GetWidth() const;
        uint32_t GetHeight() const;
        uint32_t GetDepth() const;
        uint32_t GetMipLevels() const;
        uint32_t GetLayerCount() const;
        Format GetFormat() const;
        ImageUsage GetUsage() const;

    private:
        friend class Device;
        friend class CommandList;
        friend class BindlessSet;
        friend class DescriptorSet;
        friend class Swapchain;

        std::unique_ptr<ImageImpl> mImpl;
    };
}// namespace moe::rhi