add_library(moe-neo STATIC
  Animation.cpp
  Assets.cpp
  Audio.cpp
  Content.cpp
  Font.cpp
  Importer.cpp
  Input.cpp
  OggProvider.cpp
  Renderer.cpp
  SceneGraph.cpp
  SwapchainImage.cpp
  Text.cpp
  TextureLoader.cpp
  Uploader.cpp
  Window.cpp
)

target_include_directories(moe-neo PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(moe-neo PUBLIC
  moe-core
  moe-rhi
  glm::glm
)

# importer (fastgltf) is a reader of the generic data model; GLFW only for Window
target_link_libraries(moe-neo PRIVATE
  fastgltf::fastgltf
  glfw
  OpenAL::OpenAL
  ogg vorbis vorbisfile
)