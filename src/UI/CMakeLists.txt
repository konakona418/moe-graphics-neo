# Declarative UI system: value-tree views (rebuilt every frame) over an
# immediate-mode recorder. Depends on moe-neo for the renderer, assets and the
# text renderer; distinct from moe-debugui (ImGui overlay).
add_library(moe-ui STATIC
  Ui.cpp
  Layout.cpp
  Render.cpp
  Style.cpp
)

target_include_directories(moe-ui PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(moe-ui PUBLIC
  moe-neo
  glm::glm
)

target_compile_definitions(moe-ui PRIVATE
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
