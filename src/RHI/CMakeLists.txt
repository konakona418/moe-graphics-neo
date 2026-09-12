add_library(moe-rhi STATIC
  Vulkan/BindlessSet.cpp
  Vulkan/Buffer.cpp
  Vulkan/CommandList.cpp
  Vulkan/DescriptorSet.cpp
  Vulkan/Device.cpp
  Vulkan/Fence.cpp
  Vulkan/Image.cpp
  Vulkan/Pipeline.cpp
  Vulkan/PipelineCache.cpp
  Vulkan/PipelineState.cpp
  Vulkan/Queue.cpp
  Vulkan/RenderGraph.cpp
  Vulkan/Sampler.cpp
  Vulkan/Shader.cpp
  Vulkan/Swapchain.cpp
  Vulkan/TimelineSemaphore.cpp
  Vulkan/VmaImpl.cpp
)

target_include_directories(moe-rhi PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(moe-rhi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Vulkan
)

target_link_libraries(moe-rhi PUBLIC
  vk-bootstrap::vk-bootstrap
  volk::volk
  VulkanMemoryAllocator
)

# moe-core only for the spdlog-based Logger (async pool); never leaks into
# the RHI public API.
target_link_libraries(moe-rhi PRIVATE
  moe-core
  spirv-reflect-static
)