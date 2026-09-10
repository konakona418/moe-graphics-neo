add_library(moe-core STATIC
  Error.cpp
  FileIo.cpp
  Logger.cpp
)

target_include_directories(moe-core PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/vendors/stb
)

# spdlog headers expose fmt headers; without fmt::fmt PUBLIC, consumers pick up
# a system fmt of a different version and break the link.
target_link_libraries(moe-core PUBLIC
  spdlog::spdlog
  fmt::fmt
)
