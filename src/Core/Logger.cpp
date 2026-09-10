#include "Core/Logger.hpp"

#include <mutex>
#include <spdlog/async.h>
#include <spdlog/async_logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace moe {
    std::shared_ptr<Logger> Logger::mInstance{nullptr};

    void Logger::initialize() {
        constexpr std::size_t queue_size = 8192;
        spdlog::init_thread_pool(queue_size, 1);

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%T] [%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("engine.log", true);
        file_sink->set_pattern("[%Y-%m-%d %T] [%l] %v");

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};

        mLogger = std::make_shared<spdlog::async_logger>(
                "moe",
                sinks.begin(), sinks.end(),
                spdlog::thread_pool(),
                spdlog::async_overflow_policy::block);
        mLogger->set_level(spdlog::level::debug);
        mLogger->flush_on(spdlog::level::info);

        spdlog::register_logger(mLogger);
    }

    void Logger::shutdown() {
        spdlog::drop_all();
        mLogger.reset();
        spdlog::shutdown();
    }

    void Logger::flush() {
        if (mLogger) mLogger->flush();
    }

    void Logger::setThreadName(std::string_view name) {
        static std::mutex mutex;
        // protect mThreadNames map
        {
            std::lock_guard<std::mutex> lk(mutex);
            auto logger = get();
            logger->mThreadNames[std::this_thread::get_id()] = name;
        }
    }

    std::shared_ptr<Logger> Logger::get() {
        static std::once_flag flag;
        std::call_once(flag, []() {
            mInstance = std::make_shared<Logger>();
            mInstance->initialize();
        });
        return mInstance;
    }
}// namespace moe