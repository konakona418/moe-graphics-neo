add_library(moe-example-common STATIC
  common/App.cpp
)

target_include_directories(moe-example-common PUBLIC
  ${PROJECT_SOURCE_DIR}
)

target_link_libraries(moe-example-common PUBLIC
  moe-neo
  moe-debugui
  glfw
)

add_executable(moe-example-forward
  forward/main.cpp
)

target_link_libraries(moe-example-forward PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-forward PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-forward moe-shaders)
endif()

add_executable(moe-example-grass
  grass/main.cpp
)

target_link_libraries(moe-example-grass PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-grass PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-grass moe-shaders)
endif()

add_executable(moe-example-clouds
  clouds/main.cpp
)

target_link_libraries(moe-example-clouds PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-clouds PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-clouds moe-shaders)
endif()

add_executable(moe-example-forwardplus
  forwardplus/main.cpp
)

target_link_libraries(moe-example-forwardplus PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-forwardplus PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-forwardplus moe-shaders)
endif()

add_executable(moe-example-textured
  textured/main.cpp
)

target_link_libraries(moe-example-textured PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-textured PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-textured moe-shaders)
endif()

add_executable(moe-example-snow
  snow/main.cpp
)

target_link_libraries(moe-example-snow PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-snow PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-snow moe-shaders)
endif()