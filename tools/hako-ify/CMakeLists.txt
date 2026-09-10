cmake_minimum_required(VERSION 3.10.0)
project(hakoify VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "----------------------------------------")
message(STATUS "📦 Hako-ify resource packaging tool for moe-graphics")
message(STATUS "Version ${PROJECT_VERSION}")
message(STATUS "----------------------------------------")

# disable std-exceptions and rtti
if (MSVC)
    add_compile_options(/GS- /EHs-c-)
else()
    add_compile_options(-fno-exceptions -fno-rtti)
endif()

add_executable(
    hako-ify 
    main.cpp
)