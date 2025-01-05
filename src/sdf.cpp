#include "sdf.h"
#include <fstream>
#include <string>
#include <sstream>
#include "log.h"
#include "graphics_context.h"
#include "buffer.h"
#include "vk_helpers.h"

bool sdf_load_from_file(SDF& out_sdf, const char* filepath)
{
    std::fstream f(filepath);
    if (!f.is_open())
    {
        LOG_ERROR("Failed to open file '%s'", filepath);
        return false;
    }

    std::string line;


    std::vector<uint32_t> dims;
    std::vector<float> origins;
    float grid_spacing = 0.0f;
    std::vector<float> distance_values;

    if (!std::getline(f, line)) { LOG_ERROR("Invalid file!"); return false; }
    // Read dims
    {
        std::stringstream ss(line);
        for (std::string token; std::getline(ss, token, ' '); )
        {
            uint32_t d = std::atoi(token.c_str());
            assert(d != 0);
            dims.push_back(d);
        }

        assert(dims.size() == 3);
    }

    if (!std::getline(f, line)) { LOG_ERROR("Invalid file!"); return false; }
    // Read origin
    {
        std::stringstream ss(line);
        for (std::string token; std::getline(ss, token, ' '); )
        {
            float o = std::atof(token.c_str());
            origins.push_back(o);
        }

        assert(origins.size() == 3);
    }

    if (!std::getline(f, line)) { LOG_ERROR("Invalid file!"); return false; }
    // Read grid spacing
    grid_spacing = std::atof(line.c_str());

    assert(grid_spacing > 0.0f);

    // Read values
    while (std::getline(f, line))
    {
        distance_values.push_back(std::atof(line.c_str()));
    }

    uint32_t total_grid_size = dims[0] * dims[1] * dims[2];
    assert(distance_values.size() == total_grid_size);

    out_sdf.dims = glm::uvec3(dims[0], dims[1], dims[2]);
    out_sdf.grid_origin = glm::vec3(origins[0], origins[1], origins[2]);
    out_sdf.grid_spacing = grid_spacing;
    out_sdf.data = std::move(distance_values);

    return true;
}

bool SDF::init_texture(Context& ctx)
{
    ctx.create_texture(texture, dims.x, dims.y, dims.z, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TYPE_3D, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    size_t buffer_size = dims.x * dims.y * dims.z * sizeof(float);
    BufferDesc desc{};
    desc.size = buffer_size;
    desc.usage_flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    desc.data = data.data();
    Buffer staging_buffer = ctx.create_buffer(desc);

    VkCommandBuffer cmd = ctx.allocate_and_begin_command_buffer();

    {
        VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            0,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            0,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            texture.image
        );

        VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep_info.imageMemoryBarrierCount = 1;
        dep_info.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    VkBufferImageCopy2 region{ VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2 };
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { dims.x, dims.y, dims.z };
    region.imageOffset = { 0, 0, 0 };

    VkCopyBufferToImageInfo2 copy_image{ VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2 };
    copy_image.srcBuffer = staging_buffer.buffer;
    copy_image.dstImage = texture.image;
    copy_image.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copy_image.regionCount = 1;
    copy_image.pRegions = &region;

    vkCmdCopyBufferToImage2(cmd, &copy_image);

    { // Transition to read only optimal
        VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            0,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            0,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            texture.image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, VK_REMAINING_MIP_LEVELS
        );

        VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep_info.imageMemoryBarrierCount = 1;
        dep_info.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmd;

    VK_CHECK(vkQueueSubmit(ctx.transfer_queue, 1, &info, VK_NULL_HANDLE));

    VK_CHECK(vkQueueWaitIdle(ctx.transfer_queue));

    VK_CHECK(vkResetCommandPool(ctx.device, ctx.transfer_command_pool, 0));

    vmaDestroyBuffer(ctx.allocator, staging_buffer.buffer, staging_buffer.allocation);

    texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    return true;
}
