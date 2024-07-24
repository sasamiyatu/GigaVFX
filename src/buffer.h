#pragma once
#include "defines.h"
#include "vma/vk_mem_alloc.h"

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

    operator bool() const { return buffer != VK_NULL_HANDLE; }
};