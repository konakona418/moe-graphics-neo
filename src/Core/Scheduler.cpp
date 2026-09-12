#include "Core/Scheduler.hpp"
#include <Core/Profile.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace moe {
    struct Scheduler::Impl {
        std::vector<std::thread> mWorkers;
        std::mutex mJobMutex;
        std::condition_variable mJobCv;
        std::deque<std::function<void()>> mJobs;
        bool mStopping{false};

        std::mutex mMainMutex;
        std::deque<std::function<void()>> mMainJobs;

        bool mRunning{false};
        uint32_t mWorkerCount{0};

        void WorkerLoop() {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lock(mJobMutex);
                    mJobCv.wait(lock, [this] { return mStopping || !mJobs.empty(); });
                    if (mStopping && mJobs.empty()) {
                        return;
                    }
                    job = std::move(mJobs.front());
                    mJobs.pop_front();
                }
                if (job) {
                    job();
                }
            }
        }
    };

    Scheduler::Scheduler() = default;

    Scheduler::~Scheduler() {
        Shutdown();
    }

    bool Scheduler::Init(uint32_t workerCount) {
        MOE_PROFILE_ZONE();
        if (mImpl && mImpl->mRunning) {
            return false;
        }
        if (mImpl == nullptr) {
            mImpl = std::make_unique<Impl>();
        }
        if (workerCount == 0) {
            const uint32_t hardware = std::thread::hardware_concurrency();
            workerCount = hardware > 1 ? hardware - 1 : 1;
        }
        mImpl->mWorkerCount = workerCount;
        mImpl->mStopping = false;
        mImpl->mWorkers.reserve(workerCount);
        for (uint32_t i = 0; i < workerCount; ++i) {
            mImpl->mWorkers.emplace_back([impl = mImpl.get()] { impl->WorkerLoop(); });
        }
        mImpl->mRunning = true;
        return true;
    }

    void Scheduler::Shutdown() {
        MOE_PROFILE_ZONE();
        if (mImpl == nullptr || !mImpl->mRunning) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mImpl->mJobMutex);
            mImpl->mStopping = true;
        }
        mImpl->mJobCv.notify_all();
        for (auto& worker : mImpl->mWorkers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        mImpl->mWorkers.clear();
        mImpl->mRunning = false;
        mImpl->mWorkerCount = 0;
    }

    bool Scheduler::IsRunning() const {
        return mImpl != nullptr && mImpl->mRunning;
    }

    void Scheduler::Submit(std::function<void()> job) {
        if (mImpl == nullptr || !job) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mImpl->mJobMutex);
            if (mImpl->mStopping) {
                return;
            }
            mImpl->mJobs.push_back(std::move(job));
        }
        mImpl->mJobCv.notify_one();
    }

    void Scheduler::PostToMain(std::function<void()> job) {
        if (mImpl == nullptr || !job) {
            return;
        }
        std::lock_guard<std::mutex> lock(mImpl->mMainMutex);
        mImpl->mMainJobs.push_back(std::move(job));
    }

    void Scheduler::Pump() {
        if (mImpl == nullptr) {
            return;
        }
        // Swap first so jobs posted while pumping run on the next Pump, keeping
        // a single frame bounded.
        std::deque<std::function<void()>> jobs;
        {
            std::lock_guard<std::mutex> lock(mImpl->mMainMutex);
            jobs.swap(mImpl->mMainJobs);
        }
        for (auto& job : jobs) {
            if (job) {
                job();
            }
        }
    }

    uint32_t Scheduler::GetWorkerCount() const {
        return mImpl != nullptr ? mImpl->mWorkerCount : 0;
    }
}// namespace moe
