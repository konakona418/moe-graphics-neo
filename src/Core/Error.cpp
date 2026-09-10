#include "Core/Error.hpp"

#include <mutex>

namespace moe {
    namespace {
        std::mutex gErrorMutex;
        std::string gLastError;
    }// namespace

    void Error::Set(std::string message) {
        const std::lock_guard<std::mutex> lock(gErrorMutex);
        gLastError = std::move(message);
    }

    std::string Error::Get() {
        const std::lock_guard<std::mutex> lock(gErrorMutex);
        return gLastError;
    }

    void Error::Clear() {
        const std::lock_guard<std::mutex> lock(gErrorMutex);
        gLastError.clear();
    }
}// namespace moe
