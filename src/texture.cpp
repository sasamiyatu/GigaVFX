#include "texture.h"
#include "stb_image.h"
#include "vma/vk_mem_alloc.h"

bool load_texture_from_file(const char* filepath, Texture& tex)
{
    int x, y, c;
    stbi_uc* data = stbi_load(filepath, &x, &y, &c, 4);
    if (!data)
    {
        return false;
    }

    tex.source = data;
    tex.width = x;
    tex.height = x;

    tex.name = strdup(filepath);

    return true;
}

void Texture::destroy(VkDevice device, VmaAllocator allocator)
{
    vkDestroyImageView(device, view, nullptr);
    vmaDestroyImage(allocator, image, allocation);
}
