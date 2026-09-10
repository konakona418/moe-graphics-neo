# ImGui debug overlay (migrated from the old engine's initImGUI). The name
# moe-debugui avoids the existing moe-ui (old widget system, kept for the old
# engine); they can merge once the game verdict removes one of them.
add_library(moe-debugui STATIC
  DebugUI.cpp
  Im3dDrawer.cpp
)

target_include_directories(moe-debugui PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/vendors/imgui
  ${PROJECT_SOURCE_DIR}/vendors/im3d
)

target_include_directories(moe-debugui PRIVATE
  ${PROJECT_SOURCE_DIR}/vendors/imgui/backends
)

target_link_libraries(moe-debugui PUBLIC
  moe-rhi
  glm::glm
  glfw
)

target_link_libraries(moe-debugui PRIVATE
  volk::volk
  moe-core
)

target_compile_definitions(moe-debugui PRIVATE
  IMGUI_IMPL_VULKAN_USE_VOLK # imgui_impl_vulkan talks through volk
  MOE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

# imgui + im3d sources (vendored, compiled directly like the old engine did)
target_sources(moe-debugui PRIVATE
  ${PROJECT_SOURCE_DIR}/vendors/imgui/imgui.cpp
  ${PROJECT_SOURCE_DIR}/vendors/imgui/imgui_demo.cpp
  ${PROJECT_SOURCE_DIR}/vendors/imgui/imgui_draw.cpp
  ${PROJECT_SOURCE_DIR}/vendors/imgui/imgui_tables.cpp
  ${PROJECT_SOURCE_DIR}/vendors/imgui/imgui_widgets.cpp
  ${PROJECT_SOURCE_DIR}/vendors/imgui/backends/imgui_impl_glfw.cpp
  ${PROJECT_SOURCE_DIR}/vendors/imgui/backends/imgui_impl_vulkan.cpp
  ${PROJECT_SOURCE_DIR}/vendors/im3d/im3d.cpp
)
