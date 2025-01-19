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

	inline void full_barrier(VkCommandBuffer cmd)
	{
		VkMemoryBarrier2 memory_barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
		memory_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
		memory_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT_KHR | VK_ACCESS_2_MEMORY_WRITE_BIT_KHR ;
		memory_barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
		memory_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT_KHR | VK_ACCESS_2_MEMORY_WRITE_BIT_KHR;

		VkDependencyInfo dependency_info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependency_info.memoryBarrierCount = 1;
		dependency_info.pMemoryBarriers = &memory_barrier;

		vkCmdPipelineBarrier2(cmd, &dependency_info);
	}

	inline VkDeviceAddress get_buffer_device_address(VkDevice device, VkBuffer buffer)
	{
		VkBufferDeviceAddressInfo info{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
		info.buffer = buffer;
		return vkGetBufferDeviceAddress(device, &info);
	}

	inline VkDeviceAddress  get_acceleration_structure_device_address(VkDevice device, VkAccelerationStructureKHR as)
	{
		VkAccelerationStructureDeviceAddressInfoKHR info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
		info.accelerationStructure = as;
		return vkGetAccelerationStructureDeviceAddressKHR(device, &info);
	}

	// From NVIDIA sample framework:
	// Adds a simple command that ensures that all fragment shader reads writes have finished before all
	// subsequent fragment shader reads and writes (in the current scope).
	// Note that on NV hardware, unless you need a layout transition, there's little benefit to using
	// memory barriers for each of the individual objects (and in fact may run into issues with the
	// Vulkan specification).
	// The dependency flags are BY_REGION_BIT by default, since most calls to cmdBarrier come from
	// dependencies inside render passes, which require this (according to section 6.6.1).
	inline void fragment_barrier_simple(VkCommandBuffer cmd)
	{
		const VkPipelineStageFlags stage_flags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = barrier.srcAccessMask;

		vkCmdPipelineBarrier(cmd, stage_flags, stage_flags, VK_DEPENDENCY_BY_REGION_BIT,
			1, &barrier,
			0, VK_NULL_HANDLE,
			0, VK_NULL_HANDLE);
	}

	inline void memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src_stage_flags, VkPipelineStageFlags dst_stage_flags,
		VkAccessFlags src_access, VkAccessFlags dst_access)
	{
		VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		barrier.srcAccessMask = src_access;
		barrier.dstAccessMask = dst_access;
	
		vkCmdPipelineBarrier(cmd, src_stage_flags, dst_stage_flags, 0, 1, &barrier, 0, nullptr, 0, nullptr);
	}

	inline void begin_label(VkCommandBuffer cmd, const char* name, glm::vec4 color)
	{
		VkDebugUtilsLabelEXT label_info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label_info.pLabelName = name;
		label_info.color[0] = 0.0f;
		label_info.color[1] = 0.0f;
		label_info.color[2] = 1.0f;
		label_info.color[3] = 1.0f;
		vkCmdBeginDebugUtilsLabelEXT(cmd, &label_info);
	}

	inline void end_label(VkCommandBuffer cmd)
	{
		vkCmdEndDebugUtilsLabelEXT(cmd);
	}
}