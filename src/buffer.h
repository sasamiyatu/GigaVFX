#pragma once
#include "defines.h"
#include "vma/vk_mem_alloc.h"
#include "graphics_context.h"

struct BufferDesc
{
    size_t size;
    VkBufferUsageFlags usage_flags;
    VmaAllocationCreateFlags allocation_flags;
    void* data;
};

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size = 0;

    operator bool() const { return buffer != VK_NULL_HANDLE; }
};

struct GPUBuffer
{
    Buffer staging_buffers[Context::frames_in_flight];
    Buffer gpu_buffer;

    // Handy when passing into DescriptorInfo etc
    operator VkBuffer() const { return gpu_buffer.buffer; }
};