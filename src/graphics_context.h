#pragma once

#include "defines.h"
#include "VkBootstrap.h"
#include "vma/vk_mem_alloc.h"
#include "texture.h"

struct Buffer;
struct BufferDesc;
struct GPUBuffer;

struct Context
{
    static constexpr uint32_t frames_in_flight = 2;
    SDL_Window* window;
    int window_width, window_height;
    vkb::Instance instance;
    VkSurfaceKHR surface;
    vkb::PhysicalDevice physical_device;
    vkb::Device device;
    uint32_t graphics_queue_family_index;
    VkQueue graphics_queue;
    uint32_t transfer_queue_family_index;
    VkQueue transfer_queue;
    vkb::Swapchain swapchain;

    VmaAllocator allocator;

    struct
    {
        VkSampler bilinear;
        VkSampler point;
        VkSampler bilinear_clamp;
    } samplers;

    std::vector<Texture> swapchain_textures;
    uint32_t swapchain_image_index = 0;

    VkCommandPool transfer_command_pool;
    VkCommandBuffer transfer_command_buffer;

    VkDescriptorPool imgui_descriptor_pool;

    VkDescriptorPool bindless_descriptor_pool;
    VkDescriptorSetLayout bindless_descriptor_set_layout;
    VkDescriptorSet bindless_descriptor_set;

    uint32_t frame_index = 0;
    VkCommandPool command_pools[frames_in_flight];
    VkCommandBuffer command_buffers[frames_in_flight];
    VkFence frame_fences[frames_in_flight];
    VkSemaphore image_acquired_semaphore[frames_in_flight];
    VkSemaphore rendering_finished_semaphore[frames_in_flight];
    VkQueryPool query_pool[frames_in_flight];

    struct radix_sort_vk* radix_sort_instance = nullptr;

    double smoothed_frame_time_ns = 0.0;
    uint64_t frames_rendered = 0;

    void init(int window_width, int window_height);

    void shutdown();

    VkCommandBuffer begin_frame();

    void end_frame(VkCommandBuffer command_buffer);

    inline Texture& get_swapchain_texture() { return swapchain_textures[swapchain_image_index]; }

    bool create_texture(Texture& texture, uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageType image_type, VkImageUsageFlags usage, uint32_t mip_levels = 1, uint32_t array_layers = 1);

    bool create_textures(Texture* textures, uint32_t count);

    Buffer create_buffer(const BufferDesc& desc, size_t alignment = 0);
    void destroy_buffer(Buffer& buffer);

    GPUBuffer create_gpu_buffer(const BufferDesc& desc, size_t alignment = 0);
    void destroy_buffer(GPUBuffer& buffer);
    void map_buffer(const GPUBuffer& buffer, void** mapped);
    void unmap_buffer(const GPUBuffer& buffer);
    void upload_buffer(const GPUBuffer& buffer, VkCommandBuffer cmd, uint32_t offset = 0, uint32_t size = 0);

    VkCommandBuffer allocate_and_begin_command_buffer();
    void end_command_buffer_submit_and_free(VkCommandBuffer cmd);

    VkDeviceAddress buffer_device_address(const Buffer& buffer);
};

