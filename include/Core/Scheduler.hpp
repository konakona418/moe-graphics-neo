#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace moe {
    // Generic, GPU-agnostic task scheduler: a fixed pool of worker threads for
    // CPU jobs, plus a main-thread task queue drained by Pump(). GPU completion
    // waits are serviced by their own thread (see neo::TransferContext); this
    // class never blocks on the GPU.
    //
    // Lifecycle is explicit: Init() then Shutdown() (the destructor aborts if
    // still running, mirroring the RHI leak trap).
    class Scheduler {
    public:
        Scheduler();
        ~Scheduler();

        Scheduler(const Scheduler&) = delete;
        Scheduler& operator=(const Scheduler&) = delete;

        // Starts the pool. workerCount == 0 selects max(1, hardware_concurrency
        // - 1). Idempotent-safe: calling Init twice fails.
        bool Init(uint32_t workerCount = 0);

        // Stops and joins all workers. Idempotent.
        void Shutdown();

        bool IsRunning() const;

        // Runs `job` on a worker thread.
        void Submit(std::function<void()> job);

        // Runs `job` on the thread that calls Pump() (the main thread). Safe to
        // call from any thread, including worker/completion threads.
        void PostToMain(std::function<void()> job);

        // Runs every main-thread job queued so far. Call once per frame.
        void Pump();

        uint32_t GetWorkerCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe
