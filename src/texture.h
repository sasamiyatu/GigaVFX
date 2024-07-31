#pragma once
#include "defines.h"
#include "vma/vk_mem_alloc.h"

struct Texture
{
    uint8_t* source;
    int width;
    int height;

    const char* name;

    VkFormat format;
    VkImage image;
    VkImageView view;
    VkImageLayout layout;
    VmaAllocation allocation;
    VkDescriptorSet descriptor_set; // Used with imgui

    void destroy(VkDevice device, VmaAllocator allocator);
};

inline VkImageAspectFlagBits determine_image_aspect(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

bool load_texture_from_file(const char* filepath, Texture& texture);