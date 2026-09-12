add_library(moe-example-common STATIC
  common/App.cpp
  common/Bloom.cpp
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

add_executable(moe-example-toon
  toon/main.cpp
)

target_link_libraries(moe-example-toon PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-toon PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-toon moe-shaders)
endif()

add_executable(moe-example-halftone
  halftone/main.cpp
)

target_link_libraries(moe-example-halftone PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-halftone PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-halftone moe-shaders)
endif()

add_executable(moe-example-text
  text/main.cpp
)

target_link_libraries(moe-example-text PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-text PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-text moe-shaders)
endif()

add_executable(moe-example-nubis
  nubis/main.cpp
)

target_link_libraries(moe-example-nubis PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-nubis PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-nubis moe-shaders)
endif()

add_executable(moe-example-toon-cloud
  toon_cloud/main.cpp
)

target_link_libraries(moe-example-toon-cloud PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-toon-cloud PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-toon-cloud moe-shaders)
endif()

add_executable(moe-example-ui
  ui/main.cpp
)

target_link_libraries(moe-example-ui PRIVATE
  moe-example-common
  moe-ui
)

target_compile_definitions(moe-example-ui PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-ui moe-shaders)
endif()

add_executable(moe-example-hakoniwa
  hakoniwa/src/main.cpp
  hakoniwa/src/Terrain.cpp
  hakoniwa/src/Sky.cpp
  hakoniwa/src/Grass.cpp
  hakoniwa/src/Physics.cpp
  hakoniwa/src/Post.cpp
  hakoniwa/src/Minimap.cpp
  hakoniwa/src/Hud.cpp
)

target_link_libraries(moe-example-hakoniwa PRIVATE
  moe-example-common
  moe-ui
)

target_compile_definitions(moe-example-hakoniwa PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-hakoniwa moe-shaders)
endif()

add_executable(moe-example-fluid2d
  fluid2d/main.cpp
)

target_link_libraries(moe-example-fluid2d PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-fluid2d PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-fluid2d moe-shaders)
endif()

add_executable(moe-example-fluid3d
  fluid3d/main.cpp
)

target_link_libraries(moe-example-fluid3d PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-fluid3d PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-fluid3d moe-shaders)
endif()
add_executable(moe-example-async-readback
  async_readback/main.cpp
)

target_link_libraries(moe-example-async-readback PRIVATE
  moe-example-common
)

target_compile_definitions(moe-example-async-readback PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(TARGET moe-shaders)
    add_dependencies(moe-example-async-readback moe-shaders)
endif()
