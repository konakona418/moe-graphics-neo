#include <Core/AsyncEvent.hpp>
#include <Core/Scheduler.hpp>
#include <Core/Task.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(cond)                                               \
    do {                                                          \
        if (!(cond)) {                                            \
            std::fprintf(stderr, "Scheduler FAILED: %s (%d)\n",   \
                    #cond, __LINE__);                             \
            return EXIT_FAILURE;                                  \
        }                                                         \
    } while (false)

namespace {
    moe::Task<int> Add(int a, int b) {
        co_return a + b;
    }

    moe::Task<int> AddViaAwait(int a, int b) {
        const int sum = co_await Add(a, b);
        co_return sum + 1;
    }

    moe::Task<int> AwaitEvent(moe::AsyncEvent<int>& event) {
        const int value = co_await event;
        co_return value * 2;
    }
}// namespace

int main() {
    moe::Scheduler scheduler;
    CHECK(scheduler.Init(2));
    CHECK(scheduler.IsRunning());
    CHECK(scheduler.GetWorkerCount() == 2);

    // ---- worker jobs all run ----
    std::atomic<int> count{0};
    for (int i = 0; i < 100; ++i) {
        scheduler.Submit([&count] { count.fetch_add(1); });
    }
    for (int spin = 0; spin < 2000 && count.load() < 100; ++spin) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(count.load() == 100);

    // ---- PostToMain only runs on Pump ----
    bool ran = false;
    scheduler.PostToMain([&ran] { ran = true; });
    CHECK(!ran);
    scheduler.Pump();
    CHECK(ran);

    // ---- Task chaining (co_await of a Task) ----
    {
        moe::Task<int> task = AddViaAwait(20, 21);
        task.Start();
        CHECK(task.IsDone());
        CHECK(task.Result() == 42);
    }

    // ---- AsyncEvent set from a worker resumes the coroutine on main ----
    {
        moe::AsyncEvent<int> event(scheduler);
        moe::Task<int> task = AwaitEvent(event);
        task.Start();
        CHECK(!task.IsDone());
        scheduler.Submit([&event] { event.SetValue(21); });
        for (int spin = 0; spin < 2000 && !task.IsDone(); ++spin) {
            scheduler.Pump();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(task.IsDone());
        CHECK(task.Result() == 42);
    }

    // ---- AsyncEvent already set: awaiter does not suspend ----
    {
        moe::AsyncEvent<int> event(scheduler);
        event.SetValue(7);
        moe::Task<int> task = AwaitEvent(event);
        task.Start();
        CHECK(task.IsDone());
        CHECK(task.Result() == 14);
    }

    scheduler.Shutdown();
    CHECK(!scheduler.IsRunning());

    std::printf("Scheduler tests passed.\n");
    return EXIT_SUCCESS;
}
