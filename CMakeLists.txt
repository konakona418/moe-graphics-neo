cmake_minimum_required(VERSION 3.10.0)
project(moe-graphics VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C++20 module scanning is broken with GCC 16 (breaks openal-soft and other
# vendored code); this project does not use modules.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# moe-core's Scheduler uses std::thread/std::condition_variable.
find_package(Threads REQUIRED)

add_subdirectory(vendors/glfw)
add_subdirectory(vendors/glm)
add_subdirectory(vendors/Vulkan-Headers)
add_subdirectory(vendors/vk-bootstrap)

set(VULKAN_HEADERS_INSTALL_DIR ${PROJECT_SOURCE_DIR}/vendors/Vulkan-Headers)
add_subdirectory(vendors/VulkanMemoryAllocator)

add_subdirectory(vendors/volk)
# volk.h requires VK_NO_PROTOTYPES when including Vulkan headers.
target_compile_definitions(volk PUBLIC VK_NO_PROTOTYPES)

add_subdirectory(vendors/fmt)
set(SPDLOG_FMT_EXTERNAL ON)
add_subdirectory(vendors/spdlog)

# profiling: the Tracy client backs every MOE_PROFILE_* macro (see
# Core/Profile.hpp). On-demand means no data is recorded until the server
# connects; rebuild with -DTRACY_ON_DEMAND=OFF to capture from launch.
option(MOE_ENABLE_TRACY "Enable Tracy profiler instrumentation" ON)
add_library(moe-tracy INTERFACE)
add_library(moe::tracy ALIAS moe-tracy)
if(MOE_ENABLE_TRACY)
    set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
    set(TRACY_STATIC ON CACHE BOOL "" FORCE)
    add_subdirectory(vendors/tracy)
    target_link_libraries(moe-tracy INTERFACE Tracy::TracyClient)
endif()

# openal-soft auto-enables C++20 modules with modern GCC; the GCC module
# implementation breaks openal-soft's templates, so force it off.
set(ALSOFT_ENABLE_MODULES OFF CACHE BOOL "" FORCE)
add_subdirectory(vendors/openal-soft)
add_subdirectory(vendors/ogg)
add_subdirectory(vendors/vorbis)

# shader reflection at load time
set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(vendors/SPIRV-Reflect)

# glTF asset loading
set(FASTGLTF_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(vendors/fastgltf)

# engine modules (Core only: the old engine's Math/Audio/Physics/Render/UI
# were purged)
add_subdirectory(src/Core)

# render abstraction layer (backend-agnostic API, Vulkan backend)
add_subdirectory(src/RHI)

# engine-generic data model + asset importer + caches (format-agnostic)
add_subdirectory(src/Neo)

# ImGui debug overlay (migrated from the old engine)
add_subdirectory(src/DebugUI)

# declarative UI system (value-tree views over the neo renderer)
add_subdirectory(src/UI)

include(CTest)
enable_testing()

# windowed examples (render demos)
add_subdirectory(examples)

# shader compilation (slang -> SPIR-V)
find_program(SLANGC_EXECUTABLE slangc)
find_program(PYTHON3_EXECUTABLE python3)
if(SLANGC_EXECUTABLE AND PYTHON3_EXECUTABLE)
    add_custom_target(moe-shaders
        COMMAND ${PYTHON3_EXECUTABLE} compile_shaders.py build
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/shaders/slang
        COMMENT "Compiling slang shaders to SPIR-V"
        VERBATIM
    )
else()
    message(WARNING "slangc/python3 not found; shaders will not be compiled. the engine expects shaders/*.spv at runtime.")
endif()

add_subdirectory(test)

# tools
message(STATUS "Configuring moe-graphics utilities...")
add_subdirectory(tools/hako-ify)