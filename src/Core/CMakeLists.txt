add_library(moe-core STATIC
  Error.cpp
  FileIo.cpp
  Logger.cpp
  Scheduler.cpp
)

target_include_directories(moe-core PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/vendors/stb
)

# spdlog headers expose fmt headers; without fmt::fmt PUBLIC, consumers pick up
# a system fmt of a different version and break the link.
# moe::tracy is PUBLIC so TRACY_ENABLE (and the Tracy include dir) reach every
# module that links moe-core, directly or privately.
target_link_libraries(moe-core PUBLIC
  spdlog::spdlog
  fmt::fmt
  moe::tracy
  Threads::Threads
)
