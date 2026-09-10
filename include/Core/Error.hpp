#pragma once

#include <string>
#include <utility>

namespace moe {
    // Process-wide last-error slot. Engine code records failures here instead
    // of carrying per-object error strings; callers read the most recent
    // failure with Get(). Mutex-protected so worker threads (audio) can
    // report failures the main thread can read.
    class Error {
    public:
        Error() = delete;

        static void Set(std::string message);
        static std::string Get();
        static void Clear();
    };

    // Records the message and returns false (the common failure tail).
    inline bool Fail(std::string message) {
        Error::Set(std::move(message));
        return false;
    }
}// namespace moe
