#pragma once

#include <memory>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

namespace moe {
    class Logger {
    public:
        void Initialize();
        void Shutdown();
        void Flush();

        static void SetThreadName(std::string_view name);

        template<typename... Args>
        static void Info(const char* fmt, const Args&... args) {
            Get()->Log(spdlog::level::info, fmt, args...);
        }
        template<typename... Args>
        static void Warn(const char* fmt, const Args&... args) {
            Get()->Log(spdlog::level::warn, fmt, args...);
        }
        template<typename... Args>
        static void Error(const char* fmt, const Args&... args) {
            Get()->Log(spdlog::level::err, fmt, args...);
        }
        template<typename... Args>
        static void Critical(const char* fmt, const Args&... args) {
            Get()->Log(spdlog::level::critical, fmt, args...);
        }
        template<typename... Args>
        static void Debug(const char* fmt, const Args&... args) {
            Get()->Log(spdlog::level::debug, fmt, args...);
        }

        static std::shared_ptr<Logger> Get();

    private:
        std::string_view GetThreadName() const {
            auto it = mThreadNames.find(std::this_thread::get_id());
            if (it != mThreadNames.end()) return it->second;
            return "Unknown";
        }

        template<typename... Args>
        void Log(spdlog::level::level_enum lvl, const char* fmt, Args&&... args) {
            auto threadName = GetThreadName();
            auto output = fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...);
            if (mLogger) mLogger->log(lvl, "[{}] {}", threadName, output);
        }

        std::unordered_map<std::thread::id, std::string> mThreadNames;

        std::shared_ptr<spdlog::logger> mLogger;
        static std::shared_ptr<Logger> mInstance;
    };
}// namespace moe