#include "RHI/RenderGraph.hpp"
#include <Core/Profile.hpp>

#include "Core/Error.hpp"
#include "RHI/Buffer.hpp"
#include "RHI/CommandList.hpp"
#include "RHI/Image.hpp"
#include "RhiInternal.hpp"

#include <algorithm>

namespace moe::rhi {
    struct RenderGraph::Impl {
        struct ResourceEntry {
            Image* mImage{nullptr};
            Buffer* mBuffer{nullptr};
        };
        struct PlannedBarrier {
            ResourceId mResource{kInvalidResourceId};
            bool mIsImage{false};
            ImageLayout mSrcLayout{ImageLayout::kUndefined};
            ImageLayout mDstLayout{ImageLayout::kUndefined};
            SyncInfo mSync;
        };
        struct PassEntry {
            std::string mName;
            Pass* mPass{nullptr};
            std::vector<ResourceAccess> mReads;
            std::vector<ResourceAccess> mWrites;
            std::vector<PlannedBarrier> mBarriers;
        };
        struct AccessInfo {
            bool mIsWrite{false};
            PipelineStage mStage{PipelineStage::kComputeShader};
            Access mAccess{Access::kShaderRead};
            ImageLayout mLayout{ImageLayout::kUndefined};
        };

        std::vector<ResourceEntry> mResources;
        std::vector<PassEntry> mPasses;
        std::vector<uint32_t> mExecutionOrder;
        bool mCompiled{false};
    };

    RenderGraph::RenderGraph() : mImpl(std::make_unique<Impl>()) {}

    RenderGraph::~RenderGraph() = default;

    ResourceId RenderGraph::RegisterImage(Image& image) {
        MOE_PROFILE_ZONE();
        mImpl->mResources.push_back({&image, nullptr});
        return static_cast<ResourceId>(mImpl->mResources.size() - 1);
    }

    ResourceId RenderGraph::RegisterBuffer(Buffer& buffer) {
        MOE_PROFILE_ZONE();
        mImpl->mResources.push_back({nullptr, &buffer});
        return static_cast<ResourceId>(mImpl->mResources.size() - 1);
    }

    bool RenderGraph::AddPass(const PassDesc& desc) {
        MOE_PROFILE_ZONE();
        if (desc.mPass == nullptr) {
            return false;
        }
        const auto isValid = [this](const ResourceAccess& access) {
            return access.mResource < mImpl->mResources.size();
        };
        for (const auto& access : desc.mReads) {
            if (!isValid(access)) {
                return false;
            }
        }
        for (const auto& access : desc.mWrites) {
            if (!isValid(access)) {
                return false;
            }
        }

        Impl::PassEntry entry;
        entry.mName = desc.mName != nullptr ? desc.mName : "";
        entry.mPass = desc.mPass;
        entry.mReads = desc.mReads;
        entry.mWrites = desc.mWrites;
        mImpl->mPasses.push_back(std::move(entry));
        mImpl->mCompiled = false;
        return true;
    }

    bool RenderGraph::Compile() {
        MOE_PROFILE_ZONE();
        const uint32_t passCount = static_cast<uint32_t>(mImpl->mPasses.size());

        // ---- topological sort ----
        // Edge a -> b means pass a must execute before pass b: some resource is
        // accessed by both and at least one of the two accesses is a write.
        std::vector<std::vector<uint32_t>> dependents(passCount);
        std::vector<uint32_t> indegree(passCount, 0);

        for (uint32_t resource = 0; resource < mImpl->mResources.size(); ++resource) {
            std::vector<std::pair<uint32_t, bool>> accessors; // (passIndex, isWrite)
            for (uint32_t p = 0; p < passCount; ++p) {
                bool writes = false;
                bool reads = false;
                for (const auto& a : mImpl->mPasses[p].mWrites) {
                    if (a.mResource == resource) {
                        writes = true;
                    }
                }
                for (const auto& a : mImpl->mPasses[p].mReads) {
                    if (a.mResource == resource) {
                        reads = true;
                    }
                }
                if (writes || reads) {
                    accessors.push_back({p, writes});
                }
            }
            for (size_t i = 0; i < accessors.size(); ++i) {
                for (size_t j = i + 1; j < accessors.size(); ++j) {
                    const uint32_t passA = accessors[i].first;
                    const uint32_t passB = accessors[j].first;
                    const bool aWrites = accessors[i].second;
                    const bool bWrites = accessors[j].second;
                    if (aWrites && bWrites) {
                        // two writers: order by add order to break the tie
                        dependents[passA].push_back(passB);
                        indegree[passB]++;
                    } else if (aWrites && !bWrites) {
                        // A writes R, B reads R: writer (producer) before reader
                        dependents[passA].push_back(passB);
                        indegree[passB]++;
                    } else if (!aWrites && bWrites) {
                        // A reads R, B writes R: reader must finish before the write
                        dependents[passB].push_back(passA);
                        indegree[passA]++;
                    }
                }
            }
        }

        std::vector<uint32_t> ready;
        for (uint32_t p = 0; p < passCount; ++p) {
            if (indegree[p] == 0) {
                ready.push_back(p);
            }
        }
        std::vector<uint32_t> order;
        while (!ready.empty()) {
            const uint32_t p = ready.back();
            ready.pop_back();
            order.push_back(p);
            for (const uint32_t dependent : dependents[p]) {
                if (--indegree[dependent] == 0) {
                    ready.push_back(dependent);
                }
            }
        }
        if (order.size() != passCount) {
            moe::Error::Set("RenderGraph: dependency cycle detected");
            return false;
        }

        // ---- barrier planning ----
        struct TrackedAccess {
            bool mTouched{false};
            Impl::AccessInfo mAccess;
        };
        std::vector<TrackedAccess> lastAccess(mImpl->mResources.size());

        auto planForPass = [&](uint32_t passIndex) {
            // merge the pass's accesses per resource (a write wins over a read)
            std::vector<std::pair<uint32_t, Impl::AccessInfo>> thisAccess;
            for (const auto& a : mImpl->mPasses[passIndex].mReads) {
                thisAccess.push_back({a.mResource, {false, a.mStage, a.mAccess, a.mLayout}});
            }
            for (const auto& a : mImpl->mPasses[passIndex].mWrites) {
                thisAccess.push_back({a.mResource, {true, a.mStage, a.mAccess, a.mLayout}});
            }

            for (const auto& [resource, info] : thisAccess) {
                TrackedAccess& tracked = lastAccess[resource];
                const bool needBarrier = tracked.mTouched
                        && (tracked.mAccess.mIsWrite || info.mIsWrite);
                const bool firstTouchWrite = !tracked.mTouched && info.mIsWrite
                        && mImpl->mResources[resource].mImage != nullptr;

                if (needBarrier) {
                    Impl::PlannedBarrier barrier;
                    barrier.mResource = resource;
                    barrier.mIsImage = mImpl->mResources[resource].mImage != nullptr;
                    barrier.mSrcLayout = tracked.mAccess.mLayout;
                    barrier.mDstLayout = info.mLayout;
                    barrier.mSync.mSrcStage = tracked.mAccess.mStage;
                    barrier.mSync.mSrcAccess = tracked.mAccess.mAccess;
                    barrier.mSync.mDstStage = info.mStage;
                    barrier.mSync.mDstAccess = info.mAccess;
                    mImpl->mPasses[passIndex].mBarriers.push_back(std::move(barrier));
                } else if (firstTouchWrite) {
                    // first write to an image: transition it out of Undefined
                    Impl::PlannedBarrier barrier;
                    barrier.mResource = resource;
                    barrier.mIsImage = true;
                    barrier.mSrcLayout = ImageLayout::kUndefined;
                    barrier.mDstLayout = info.mLayout;
                    barrier.mSync.mSrcStage = PipelineStage::kTopOfPipe;
                    barrier.mSync.mSrcAccess = Access::kNone;
                    barrier.mSync.mDstStage = info.mStage;
                    barrier.mSync.mDstAccess = info.mAccess;
                    mImpl->mPasses[passIndex].mBarriers.push_back(std::move(barrier));
                }

                tracked.mTouched = true;
                tracked.mAccess = info;
            }
        };

        for (const uint32_t passIndex : order) {
            planForPass(passIndex);
        }

        mImpl->mExecutionOrder = std::move(order);
        mImpl->mCompiled = true;
        return true;
    }

    bool RenderGraph::Execute(CommandList& cmd) {
        MOE_PROFILE_ZONE();
        if (!mImpl->mCompiled) {
            return false;
        }
        for (const uint32_t passIndex : mImpl->mExecutionOrder) {
            for (const auto& barrier : mImpl->mPasses[passIndex].mBarriers) {
                const Impl::ResourceEntry& resource = mImpl->mResources[barrier.mResource];
                if (barrier.mIsImage) {
                    cmd.ImageBarrier(*resource.mImage, barrier.mSrcLayout, barrier.mDstLayout, barrier.mSync);
                } else {
                    cmd.BufferBarrier(*resource.mBuffer, barrier.mSync);
                }
            }
            if (!mImpl->mPasses[passIndex].mPass->Execute(cmd)) {
                return false;
            }
        }
        return true;
    }
}// namespace moe::rhi