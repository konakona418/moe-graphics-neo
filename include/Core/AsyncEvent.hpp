#pragma once

#include <Core/Scheduler.hpp>

#include <coroutine>
#include <mutex>
#include <optional>
#include <utility>

namespace moe {
    // One-shot, thread-safe completion slot. A worker/completion thread calls
    // SetValue; the main thread either polls (TryTake) or co_awaits. Awaiters
    // always resume on the scheduler's main thread (via PostToMain), so a
    // coroutine that awaited GPU work continues where GPU/UI work is legal.
    //
    // T must be movable. The first SetValue wins; later calls are ignored.
    template<typename T>
    class AsyncEvent {
    public:
        explicit AsyncEvent(Scheduler& scheduler) : mScheduler(&scheduler) {}

        AsyncEvent(const AsyncEvent&) = delete;
        AsyncEvent& operator=(const AsyncEvent&) = delete;

        // Safe from any thread.
        void SetValue(T value) {
            std::coroutine_handle<> resume{nullptr};
            {
                std::lock_guard<std::mutex> lock(mMutex);
                if (mValue.has_value()) {
                    return;
                }
                mValue.emplace(std::move(value));
                resume = std::exchange(mContinuation, nullptr);
            }
            if (resume) {
                mScheduler->PostToMain([resume] { resume.resume(); });
            }
        }

        bool IsReady() const {
            std::lock_guard<std::mutex> lock(mMutex);
            return mValue.has_value();
        }

        // Moves the value out if ready; otherwise returns an empty optional.
        // Call on the main thread.
        std::optional<T> TryTake() {
            std::lock_guard<std::mutex> lock(mMutex);
            std::optional<T> out;
            out.swap(mValue);
            return out;
        }

        struct Awaiter {
            AsyncEvent& mEvent;

            bool await_ready() const noexcept {
                return mEvent.IsReady();
            }

            void await_suspend(std::coroutine_handle<> continuation) {
                std::coroutine_handle<> resume{nullptr};
                {
                    std::lock_guard<std::mutex> lock(mEvent.mMutex);
                    if (mEvent.mValue.has_value()) {
                        // Completed between await_ready and here: resume at once.
                        resume = continuation;
                    } else {
                        mEvent.mContinuation = continuation;
                    }
                }
                if (resume) {
                    mEvent.mScheduler->PostToMain([resume] { resume.resume(); });
                }
            }

            T await_resume() {
                std::optional<T> value = mEvent.TryTake();
                return std::move(*value);
            }
        };

        Awaiter operator co_await() {
            return Awaiter{*this};
        }

    private:
        mutable std::mutex mMutex;
        std::optional<T> mValue;
        std::coroutine_handle<> mContinuation{nullptr};
        Scheduler* mScheduler{nullptr};
    };
}// namespace moe
