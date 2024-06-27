#pragma once
#include "defines.h"

namespace VkHelpers
{
	inline VkCommandPool create_command_pool(VkDevice device, uint32_t queue_family_index)
	{
		VkCommandPoolCreateInfo info{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		info.queueFamilyIndex = queue_family_index;
		VkCommandPool cmd_pool = VK_NULL_HANDLE;
		VK_CHECK(vkCreateCommandPool(device, &info, nullptr, &cmd_pool));
		return cmd_pool;
	}

	inline void begin_command_buffer(VkCommandBuffer cmd, VkCommandBufferUsageFlags flags)
	{
		VkCommandBufferBeginInfo info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		info.flags = flags;
		VK_CHECK(vkBeginCommandBuffer(cmd, &info));
	}

	inline VkImageMemoryBarrier2 image_memory_barrier2(
		VkPipelineStageFlags2		src_stage_mask,
		VkAccessFlags2				src_access_mask,
		VkPipelineStageFlags2		dst_stage_mask,
		VkAccessFlags2				dst_access_mask,
		VkImageLayout				old_layout,
		VkImageLayout				new_layout,
		VkImage						image,
		VkImageAspectFlagBits		aspect = VK_IMAGE_ASPECT_COLOR_BIT,
		uint32_t					base_mip_level = 0,
		uint32_t					level_count = 1,
		uint32_t					base_array_layer = 0,
		uint32_t					base_layer_count = 1
	)
	{
		VkImageMemoryBarrier2 image_barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };

		image_barrier.srcStageMask = src_stage_mask;
		image_barrier.srcAccessMask = src_access_mask;
		image_barrier.dstStageMask = dst_stage_mask;
		image_barrier.dstAccessMask = dst_access_mask;
		image_barrier.oldLayout = old_layout;
		image_barrier.newLayout = new_layout;
		image_barrier.image = image;
		image_barrier.subresourceRange.aspectMask = aspect;
		image_barrier.subresourceRange.baseMipLevel = base_mip_level;
		image_barrier.subresourceRange.levelCount = level_count;
		image_barrier.subresourceRange.baseArrayLayer = base_array_layer;
		image_barrier.subresourceRange.layerCount = base_layer_count;

		return image_barrier;
	}
}