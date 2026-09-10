add_executable(moe-vk-smoke
  HeadlessVulkanSmoke.cpp
)

target_link_libraries(moe-vk-smoke PRIVATE
  vk-bootstrap::vk-bootstrap
  volk::volk
)

target_compile_definitions(moe-vk-smoke PRIVATE VK_NO_PROTOTYPES)

add_test(NAME vk-smoke COMMAND moe-vk-smoke)

add_executable(moe-rhi-compute-smoke
  ComputeSmoke.cpp
)

target_link_libraries(moe-rhi-compute-smoke PRIVATE
  moe-rhi
)

target_compile_definitions(moe-rhi-compute-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-rhi-compute-smoke moe-shaders)
endif()

add_test(NAME rhi-compute-smoke COMMAND moe-rhi-compute-smoke)

add_executable(moe-rhi-pipeline-smoke
  PipelineSmoke.cpp
)

target_link_libraries(moe-rhi-pipeline-smoke PRIVATE
  moe-rhi
)

target_compile_definitions(moe-rhi-pipeline-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-rhi-pipeline-smoke moe-shaders)
endif()

add_test(NAME rhi-pipeline-smoke COMMAND moe-rhi-pipeline-smoke)

add_executable(moe-rhi-graph-smoke
  GraphSmoke.cpp
)

target_link_libraries(moe-rhi-graph-smoke PRIVATE
  moe-rhi
)

target_compile_definitions(moe-rhi-graph-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-rhi-graph-smoke moe-shaders)
endif()

add_test(NAME rhi-graph-smoke COMMAND moe-rhi-graph-smoke)

add_executable(moe-rhi-upload-smoke
  UploadSmoke.cpp
)

target_link_libraries(moe-rhi-upload-smoke PRIVATE
  moe-rhi
)

if(TARGET moe-shaders)
    add_dependencies(moe-rhi-upload-smoke moe-shaders)
endif()

add_test(NAME rhi-upload-smoke COMMAND moe-rhi-upload-smoke)

add_executable(moe-rhi-render-smoke
  RenderSmoke.cpp
)

target_link_libraries(moe-rhi-render-smoke PRIVATE
  moe-rhi
)

target_compile_definitions(moe-rhi-render-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-rhi-render-smoke moe-shaders)
endif()

add_test(NAME rhi-render-smoke COMMAND moe-rhi-render-smoke)

add_executable(moe-rhi-clouds-smoke
  CloudsSmoke.cpp
)

target_link_libraries(moe-rhi-clouds-smoke PRIVATE
  moe-rhi
)

target_compile_definitions(moe-rhi-clouds-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-rhi-clouds-smoke moe-shaders)
endif()

add_test(NAME rhi-clouds-smoke COMMAND moe-rhi-clouds-smoke)


add_executable(moe-data-struct-tests
  DataStructures.cpp
)

target_link_libraries(moe-data-struct-tests PRIVATE
  moe-core
)

add_test(NAME data-struct-tests COMMAND moe-data-struct-tests)

add_executable(moe-neo-import-smoke
  ImportSmoke.cpp
)

target_link_libraries(moe-neo-import-smoke PRIVATE
  moe-neo
)

target_compile_definitions(moe-neo-import-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_test(NAME neo-import-smoke COMMAND moe-neo-import-smoke)

add_executable(moe-neo-cache-tests
  NeoCacheTests.cpp
)

target_link_libraries(moe-neo-cache-tests PRIVATE
  moe-neo
)

add_test(NAME neo-cache-tests COMMAND moe-neo-cache-tests)

add_executable(moe-neo-upload-smoke
  UploadMeshSmoke.cpp
)

target_link_libraries(moe-neo-upload-smoke PRIVATE
  moe-neo
)

add_test(NAME neo-upload-smoke COMMAND moe-neo-upload-smoke)

add_executable(moe-neo-audio-smoke
  AudioSmoke.cpp
)

target_link_libraries(moe-neo-audio-smoke PRIVATE
  moe-neo
)

target_compile_definitions(moe-neo-audio-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_test(NAME neo-audio-smoke COMMAND moe-neo-audio-smoke)

add_executable(moe-neo-texture-smoke
  TextureSmoke.cpp
)

target_link_libraries(moe-neo-texture-smoke PRIVATE
  moe-neo
)

target_compile_definitions(moe-neo-texture-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_test(NAME neo-texture-smoke COMMAND moe-neo-texture-smoke)

add_executable(moe-rhi-bindless-smoke
  BindlessSmoke.cpp
)

target_link_libraries(moe-rhi-bindless-smoke PRIVATE
  moe-neo
)

target_compile_definitions(moe-rhi-bindless-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-rhi-bindless-smoke moe-shaders)
endif()

add_test(NAME rhi-bindless-smoke COMMAND moe-rhi-bindless-smoke)
add_executable(moe-neo-font-smoke
  FontSmoke.cpp
)

target_link_libraries(moe-neo-font-smoke PRIVATE
  moe-neo
)

target_compile_definitions(moe-neo-font-smoke PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_test(NAME neo-font-smoke COMMAND moe-neo-font-smoke)
