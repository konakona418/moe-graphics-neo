#pragma once

#include <cstdint>
#include <memory>
#include <string>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>

namespace moe::rhi {
    class Device;
    class DefaultPipelineCache;
    class CommandList;
    class Swapchain;
}

namespace moe::ui {
    struct Im3dDrawerImpl;

    // Im3d debug drawing (points/lines/triangles; lines are expanded by a
    // geometry shader). Migrated from the old engine's VulkanIm3dDriver: the
    // same three pipelines, vertex data read via buffer device address from
    // push constants (no descriptor sets), alpha blend + depth test, one
    // 64k-vertex buffer re-uploaded per frame.
    //
    // Frame flow: the demo sets Im3d::GetAppData(), calls Im3d::NewFrame(),
    // draws primitives, Im3d::EndFrame() (via the App's mDrawIm3d hook), then
    // its render pass calls UploadVertices() before BeginRendering and
    // Record() inside the pass, after the scene.
    class Im3dDrawer {
    public:
        Im3dDrawer();
        ~Im3dDrawer();

        Im3dDrawer(const Im3dDrawer&) = delete;
        Im3dDrawer& operator=(const Im3dDrawer&) = delete;

        // Explicit teardown (idempotent). The destructor aborts if the drawer
        // was initialized but not destroyed (leak trap, same discipline as RHI).
        void Destroy();

        // Builds the three pipelines (formats from the swapchain) and the
        // vertex/staging buffers. Pipelines are owned by the cache; the cache
        // must outlive the drawer.
        bool Init(moe::rhi::Device& device, moe::rhi::DefaultPipelineCache& cache,
                moe::rhi::Swapchain& swapchain, std::string& error);

        bool IsActive() const;

        // True when the current frame has primitives to draw.
        bool HasDraws() const;

        // Records staging -> vertex buffer copies + a barrier. Call before the
        // render pass that Record() draws into (transfer must be outside).
        void UploadVertices(moe::rhi::CommandList& cmd);

        // Draws the Im3d draw lists into the ACTIVE render pass. Call after
        // the scene, before EndRendering.
        void Record(moe::rhi::CommandList& cmd);

        // Current frame's camera state, set by the demo (it owns the camera).
        glm::mat4 mViewProj{1.0f};
        glm::vec2 mViewport{0.0f, 0.0f};

    private:
        std::unique_ptr<Im3dDrawerImpl> mImpl;
    };
}// namespace moe::ui
