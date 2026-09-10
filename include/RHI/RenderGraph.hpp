#pragma once

#include "RHI/RHICommon.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace moe::rhi {
    class Device;
    class CommandList;
    class Image;
    class Buffer;

    // User pass: records commands against a command list. The graph calls
    // Execute once per execution, in topological order, after recording the
    // automatically planned barriers.
    class Pass {
    public:
        virtual ~Pass() = default;
        virtual bool Execute(CommandList& cmd) = 0;
    };

    using ResourceId = uint32_t;
    constexpr ResourceId kInvalidResourceId = UINT32_MAX;

    // How one pass touches one resource. The graph derives barriers from the
    // stage/access pairs of consecutive accesses; mLayout is used for images.
    struct ResourceAccess {
        ResourceId mResource{kInvalidResourceId};
        bool mIsWrite{false};
        PipelineStage mStage{PipelineStage::kComputeShader};
        Access mAccess{Access::kShaderRead};
        ImageLayout mLayout{ImageLayout::kUndefined}; // images only
    };

    struct PassDesc {
        const char* mName{nullptr};
        Pass* mPass{nullptr};
        std::vector<ResourceAccess> mReads;
        std::vector<ResourceAccess> mWrites;
    };

    // Lightweight orchestration graph: resources are created explicitly by the
    // user; the graph only orders passes topologically and inserts the barriers
    // between consecutive accesses that involve a write.
    class RenderGraph {
    public:
        RenderGraph();
        ~RenderGraph();

        RenderGraph(const RenderGraph&) = delete;
        RenderGraph& operator=(const RenderGraph&) = delete;

        ResourceId RegisterImage(Image& image);
        ResourceId RegisterBuffer(Buffer& buffer);

        bool AddPass(const PassDesc& desc);

        // Topologically sorts passes and plans barriers. Returns false with an
        // error message on cycles or invalid references.
        bool Compile(std::string& error);

        // Records the planned barriers and each pass's Execute in order.
        bool Execute(CommandList& cmd);

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::rhi