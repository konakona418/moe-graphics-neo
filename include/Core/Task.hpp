#pragma once

#include <coroutine>
#include <exception>
#include <optional>
#include <utility>

namespace moe {
    // Minimal lazy C++20 coroutine returning a value. It does not start until
    // awaited (co_await) or resumed explicitly. Move-only; destroying a
    // suspended Task destroys the underlying coroutine frame. Exceptions are
    // not supported (the codebase is exception-free): an escaping exception
    // terminates.
    template<typename T>
    class Task {
    public:
        struct promise_type {
            std::optional<T> mValue;
            std::coroutine_handle<> mContinuation{nullptr};

            Task get_return_object() {
                return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
            }

            std::suspend_always initial_suspend() noexcept {
                return {};
            }

            struct FinalAwaiter {
                bool await_ready() const noexcept {
                    return false;
                }

                std::coroutine_handle<> await_suspend(
                        std::coroutine_handle<promise_type> handle) const noexcept {
                    const std::coroutine_handle<> continuation = handle.promise().mContinuation;
                    return continuation != nullptr ? continuation : std::noop_coroutine();
                }

                void await_resume() const noexcept {}
            };

            FinalAwaiter final_suspend() noexcept {
                return {};
            }

            void return_value(T value) {
                mValue.emplace(std::move(value));
            }

            void unhandled_exception() {
                std::terminate();
            }
        };

        Task() = default;
        explicit Task(std::coroutine_handle<promise_type> handle) : mHandle(handle) {}

        ~Task() {
            if (mHandle) {
                mHandle.destroy();
            }
        }

        Task(Task&& other) noexcept : mHandle(std::exchange(other.mHandle, nullptr)) {}

        Task& operator=(Task&& other) noexcept {
            if (this != &other) {
                if (mHandle) {
                    mHandle.destroy();
                }
                mHandle = std::exchange(other.mHandle, nullptr);
            }
            return *this;
        }

        Task(const Task&) = delete;
        Task& operator=(const Task&) = delete;

        bool IsValid() const {
            return mHandle != nullptr;
        }

        // Fire-and-forget start: resumes the coroutine from its initial
        // suspend. Observe completion with IsDone()/Result(). Do not mix
        // Start() with awaiting the same Task.
        void Start() {
            if (mHandle && !mStarted) {
                mStarted = true;
                mHandle.resume();
            }
        }

        bool IsDone() const {
            return mHandle != nullptr && mHandle.done();
        }

        // Valid only once IsDone() is true.
        T Result() {
            return std::move(*mHandle.promise().mValue);
        }

        // Awaiting starts the coroutine (if not already running) and resumes the
        // awaiting coroutine once it completes. The awaiter borrows the handle;
        // the Task must stay alive for the duration of the await.
        struct Awaiter {
            std::coroutine_handle<promise_type> mHandle;

            bool await_ready() const noexcept {
                return mHandle == nullptr || mHandle.done();
            }

            std::coroutine_handle<> await_suspend(std::coroutine_handle<> continuation) noexcept {
                mHandle.promise().mContinuation = continuation;
                return mHandle;
            }

            T await_resume() {
                return std::move(*mHandle.promise().mValue);
            }
        };

        Awaiter operator co_await() {
            return Awaiter{mHandle};
        }

    private:
        std::coroutine_handle<promise_type> mHandle{nullptr};
        bool mStarted{false};
    };
}// namespace moe
