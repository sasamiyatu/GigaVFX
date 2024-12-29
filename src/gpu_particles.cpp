#include "gpu_particles.h"
#include "pipeline.h"
#include "graphics_context.h"
#include "hot_reload.h"
#include "../shaders/shared.h"
#include "imgui/imgui.h"
#include "camera.h"
#include "vk_helpers.h"

static uint32_t get_dispatch_size(uint32_t particle_capacity)
{
	return (particle_capacity + 63) / 64;
}

void GPUParticleSystem::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity)
{
	this->ctx = ctx;
	shader_globals = globals_buffer;
	this->particle_capacity = particle_capacity;

	{ // Render pipeline
		GraphicsPipelineBuilder builder(ctx->device, true);
		builder
			.set_vertex_shader_filepath("gpu_particles.hlsl")
			.set_fragment_shader_filepath("gpu_particles.hlsl")
			.set_cull_mode(VK_CULL_MODE_NONE)
			.add_color_attachment(render_target_format)
			.set_depth_format(VK_FORMAT_D32_SFLOAT)
			.set_depth_test(VK_TRUE)
			.set_depth_write(VK_FALSE)
			.set_depth_compare_op(VK_COMPARE_OP_LESS)
			//.set_blend_preset(BlendPreset::ADDITIVE)
			.set_blend_preset(BlendPreset::ALPHA)
			.set_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

		render_pipeline = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(render_pipeline);
	}

	{ // Emit pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_emit_particles");
		particle_emit_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_emit_pipeline);
	}

	{ // Indirect dispatch size write pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_write_dispatch");
		particle_dispatch_size_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_dispatch_size_pipeline);
	} 

	{ // Simulate pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_simulate_particles");
		particle_simulate_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_simulate_pipeline);
	}

	{ // Compact pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_compact_particles");
		particle_compact_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_compact_pipeline);
	}

	{ // Debug sort pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_debug_print_sorted_particles");
		particle_debug_sort_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_debug_sort_pipeline);
	}

	{ // Particle system globals buffer
		BufferDesc desc{};
		desc.size = sizeof(GPUParticleSystemGlobals);
		desc.usage_flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		system_globals = ctx->create_buffer(desc);

		GPUParticleSystemGlobals globals{};
		globals.particle_capacity = particle_capacity;
		globals.transform = glm::mat4(1.0f);

		void* mapped;
		vmaMapMemory(ctx->allocator, system_globals.allocation, &mapped);
		memcpy(mapped, &globals, sizeof(globals));
		vmaUnmapMemory(ctx->allocator, system_globals.allocation);
	}

	{ // Particles buffer
		for (int i = 0; i < 2; ++i)
		{
			size_t particle_buffer_size = particle_capacity * sizeof(GPUParticle);
			BufferDesc desc{};
			desc.size = particle_buffer_size;
			desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			particle_buffer[i] = ctx->create_buffer(desc);
		}
	}

	{ // Particle system state buffer
		for (int i = 0; i < 2; ++i)
		{
			BufferDesc desc{};
			desc.size = sizeof(GPUParticleSystemState);
			desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			particle_system_state[i] = ctx->create_buffer(desc);
		}
	}

	{ // Indirect dispatch buffer
		BufferDesc desc{};
		desc.size = sizeof(GPUParticleIndirectData);
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		indirect_dispatch_buffer = ctx->create_buffer(desc);
	}

	{ // Query pool used for perf measurement
		VkQueryPoolCreateInfo info{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
		info.queryCount = 256;
		info.queryType = VK_QUERY_TYPE_TIMESTAMP;
		VK_CHECK(vkCreateQueryPool(ctx->device, &info, nullptr, &query_pool));
	}

	{ // Radix sort context
		//sort_context = radix_sort_context_create(ctx, particle_capacity);

		radix_sort_vk_memory_requirements memory_requirements{};
		radix_sort_vk_get_memory_requirements(ctx->radix_sort_instance, particle_capacity, &memory_requirements);

		BufferDesc desc{};
		desc.size = memory_requirements.keyvals_size;
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

		for (int i = 0; i < 2; ++i)
			sort_keyval_buffer[i] = ctx->create_buffer(desc, memory_requirements.keyvals_alignment);

		desc.allocation_flags = 0;
		desc.size = memory_requirements.internal_size;
		sort_internal_buffer = ctx->create_buffer(desc, memory_requirements.internal_alignment);

		desc.usage_flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		desc.size = memory_requirements.indirect_size;
		sort_indirect_buffer = ctx->create_buffer(desc, memory_requirements.indirect_alignment);
	}

	{ // Create accelerations structure
		VkAccelerationStructureGeometryKHR blas_geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		blas_geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
		blas_geometry.geometry.aabbs = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR };

		VkAccelerationStructureBuildGeometryInfoKHR build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		build_info.geometryCount = 1;
		build_info.pGeometries = &blas_geometry;

		uint32_t max_primitive_counts = particle_capacity;

		VkAccelerationStructureBuildSizesInfoKHR size_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&build_info,
			&max_primitive_counts,
			&size_info);

		{ // Acceleration bottom level structure buffer
			BufferDesc desc{};
			desc.size = size_info.accelerationStructureSize;
			desc.usage_flags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
			blas.acceleration_structure_buffer = ctx->create_buffer(desc);
		}

		{ // Scratch buffer
			BufferDesc desc{};
			desc.size = size_info.buildScratchSize;
			desc.usage_flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			blas.scratch_buffer = ctx->create_buffer(desc);
		}

		VkAccelerationStructureCreateInfoKHR create_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		create_info.buffer = blas.acceleration_structure_buffer.buffer;
		create_info.size = size_info.accelerationStructureSize;
		create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		VK_CHECK(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &blas.acceleration_structure));

		{ // Acceleration structure input
			BufferDesc desc{};	
			desc.size = sizeof(AABBPositions) * particle_capacity;
			desc.usage_flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			particle_aabbs = ctx->create_buffer(desc);
		}
	}

	{ // Acceleration top level structure buffer
		{ // Create acceleration structure
			VkAccelerationStructureGeometryKHR geometries{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
			geometries.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
			geometries.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
			VkAccelerationStructureBuildGeometryInfoKHR build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
			build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
			build_info.geometryCount = 1;
			build_info.pGeometries = &geometries;
			uint32_t max_primitive_counts = particle_capacity;
			VkAccelerationStructureBuildSizesInfoKHR size_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
			vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &max_primitive_counts, &size_info);

			{ // AS buffer
				BufferDesc desc{};
				desc.size = size_info.accelerationStructureSize;
				desc.usage_flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
				tlas.acceleration_structure_buffer = ctx->create_buffer(desc);
			}

			{ // Scratch
				BufferDesc desc{};
				desc.size = size_info.buildScratchSize;
				desc.usage_flags = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
				tlas.scratch_buffer = ctx->create_buffer(desc);
			}

			VkAccelerationStructureCreateInfoKHR create_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
			create_info.buffer = tlas.acceleration_structure_buffer.buffer;
			create_info.size = size_info.accelerationStructureSize;
			create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
			VK_CHECK(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &tlas.acceleration_structure));

			{ // Instances buffer
				std::vector<VkAccelerationStructureInstanceKHR> instances(particle_capacity);
				VkDeviceAddress blas_ref = VkHelpers::get_acceleration_structure_device_address(ctx->device, blas.acceleration_structure);
				for (auto& i : instances)
				{
					i.mask = 0xFF;
					i.transform.matrix[0][0] = 1.0f;
					i.transform.matrix[1][1] = 1.0f;
					i.transform.matrix[2][2] = 1.0f;
					i.accelerationStructureReference = blas_ref;
				}
				BufferDesc desc{};
				desc.size = sizeof(VkAccelerationStructureInstanceKHR) * particle_capacity;
				desc.usage_flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | 
					VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | 
					VK_BUFFER_USAGE_TRANSFER_DST_BIT |
					VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
				desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
				desc.data = instances.data();
				instances_buffer = ctx->create_buffer(desc);
			}
		}

	}
}

void GPUParticleSystem::simulate(VkCommandBuffer cmd, float dt, CameraState& camera_state)
{
	static bool first = true;
	particles_to_spawn += particle_spawn_rate * dt;

	particle_sort_axis = -camera_state.forward;

	if (!first)
	{ // TODO: This performance timing stuff is probably incorrect when multiple frames are in flight
		uint64_t query_results[4];
		vkGetQueryPoolResults(ctx->device, query_pool, 0, 4, sizeof(query_results), query_results, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		uint64_t diff = query_results[1] - query_results[0];
		double nanoseconds = ctx->physical_device.properties.limits.timestampPeriod * diff;
		performance_timings.simulate_total = glm::mix(nanoseconds, performance_timings.simulate_total, 0.95);

		uint64_t delta_render = query_results[3] - query_results[2];
		double ns_render = ctx->physical_device.properties.limits.timestampPeriod * delta_render;
		performance_timings.render_total = glm::mix(ns_render, performance_timings.render_total, 0.95);
	}

	first = false;

	vkCmdResetQueryPool(cmd, query_pool, 0, 256);

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
	if (!particles_initialized)
	{ // Zero init buffers
		for (int i = 0; i < 2; ++i)
		{
			vkCmdFillBuffer(cmd, particle_buffer[i].buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmd, particle_system_state[i].buffer, 0, VK_WHOLE_SIZE, 0);
		}

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);

		particles_initialized = true;
	}

	// All passes use the same descriptors for now
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(system_globals.buffer),
		DescriptorInfo(particle_buffer[0].buffer),
		DescriptorInfo(particle_system_state[0].buffer),
		DescriptorInfo(particle_buffer[1].buffer),
		DescriptorInfo(particle_system_state[1].buffer),
		DescriptorInfo(indirect_dispatch_buffer.buffer),
		DescriptorInfo(sort_keyval_buffer[0].buffer),
		DescriptorInfo(particle_aabbs.buffer),
		DescriptorInfo(instances_buffer.buffer),
	};

	// Likewise for push constants
	GPUParticlePushConstants push_constants{};
	push_constants.delta_time = dt;
	push_constants.particles_to_spawn = (uint32_t)particles_to_spawn;
	push_constants.particle_size = particle_size;
	push_constants.particle_color = particle_color;
	push_constants.sort_axis = particle_sort_axis;
	push_constants.blas_address = VkHelpers::get_acceleration_structure_device_address(ctx->device, blas.acceleration_structure);
	
	{ // Clear output state
		vkCmdFillBuffer(cmd, particle_system_state[1].buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, particle_buffer[1].buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, particle_aabbs.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, instances_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

	{ // Emit particles
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_emit_pipeline->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_emit_pipeline->pipeline.descriptor_update_template,
			particle_emit_pipeline->pipeline.layout, 0, descriptor_info);

		vkCmdPushConstants(cmd, particle_emit_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);
		vkCmdDispatch(cmd, get_dispatch_size(push_constants.particles_to_spawn), 1, 1);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

	{ // Write indirect dispatch size
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_dispatch_size_pipeline->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_dispatch_size_pipeline->pipeline.descriptor_update_template,
			particle_dispatch_size_pipeline->pipeline.layout, 0, descriptor_info);

		vkCmdDispatch(cmd, 1, 1, 1);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

	{ // Simulate particles
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_simulate_pipeline->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_simulate_pipeline->pipeline.descriptor_update_template,
			particle_simulate_pipeline->pipeline.layout, 0, descriptor_info);
		vkCmdPushConstants(cmd, particle_simulate_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);
		vkCmdDispatchIndirect(cmd, indirect_dispatch_buffer.buffer, 0);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

	{ // Compact particles
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_compact_pipeline->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_compact_pipeline->pipeline.descriptor_update_template,
			particle_compact_pipeline->pipeline.layout, 0, descriptor_info);
		vkCmdDispatchIndirect(cmd, indirect_dispatch_buffer.buffer, 0);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

	{ // Sort particles
		if (sort_particles)
		{
			radix_sort_vk_sort_indirect_info sort_info{};
			sort_info.key_bits = 32;
			uint32_t count_offset = offsetof(GPUParticleIndirectData, draw_cmd) + offsetof(DrawIndirectCommand, instanceCount);
			sort_info.count = { indirect_dispatch_buffer.buffer, count_offset, VK_WHOLE_SIZE };
			sort_info.keyvals_even = { sort_keyval_buffer[0].buffer, 0, VK_WHOLE_SIZE };
			sort_info.keyvals_odd = { sort_keyval_buffer[1].buffer, 0, VK_WHOLE_SIZE };
			sort_info.internal = { sort_internal_buffer.buffer, 0, VK_WHOLE_SIZE };
			sort_info.indirect = { sort_indirect_buffer.buffer, 0, VK_WHOLE_SIZE };

			VkDescriptorBufferInfo keyvals_sorted{};
			radix_sort_vk_sort_indirect(ctx->radix_sort_instance, &sort_info, ctx->device, cmd, &keyvals_sorted);

			if (keyvals_sorted.buffer != sort_keyval_buffer[0].buffer)
				std::swap(sort_keyval_buffer[0], sort_keyval_buffer[1]);
		}

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

	{ // Build bottom level acceleration structure

		VkAccelerationStructureGeometryKHR blas_geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		blas_geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
		blas_geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
		blas_geometry.geometry.aabbs.data.deviceAddress = VkHelpers::get_buffer_device_address(ctx->device, particle_aabbs.buffer);
		blas_geometry.geometry.aabbs.stride = sizeof(AABBPositions);

		VkAccelerationStructureBuildGeometryInfoKHR build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };

		build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		build_info.dstAccelerationStructure = blas.acceleration_structure;
		build_info.geometryCount = 1;
		build_info.pGeometries = &blas_geometry;
		build_info.scratchData.deviceAddress = VkHelpers::get_buffer_device_address(ctx->device, blas.scratch_buffer.buffer);

		VkAccelerationStructureBuildRangeInfoKHR build_range_info{};
		build_range_info.primitiveCount = particle_capacity;
		build_range_info.primitiveOffset = 0;
		build_range_info.firstVertex = 0;
		build_range_info.transformOffset = 0;

		VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &build_range_info;
		vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_ptr);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}


	{ // Build TLAS
		VkAccelerationStructureGeometryKHR geometries{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		geometries.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometries.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometries.geometry.instances.data.deviceAddress = VkHelpers::get_buffer_device_address(ctx->device, instances_buffer.buffer);

		VkAccelerationStructureBuildGeometryInfoKHR build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };

		build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		build_info.dstAccelerationStructure = tlas.acceleration_structure;
		build_info.geometryCount = 1;
		build_info.pGeometries = &geometries;
		build_info.scratchData.deviceAddress = VkHelpers::get_buffer_device_address(ctx->device, tlas.scratch_buffer.buffer);

		VkAccelerationStructureBuildRangeInfoKHR ri{};
		ri.primitiveCount = 1;

		VkAccelerationStructureBuildRangeInfoKHR* range_info = &ri;

		vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_info);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

#if 0
	{ // Debug sort
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_debug_sort_pipeline->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_debug_sort_pipeline->pipeline.descriptor_update_template,
			particle_debug_sort_pipeline->pipeline.layout, 0, descriptor_info);
		vkCmdPushConstants(cmd, particle_debug_sort_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);
		vkCmdDispatch(cmd, 1, 1, 1);
	}
#endif



	particles_to_spawn -= std::floor(particles_to_spawn);

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);

	// Swap buffers
	std::swap(particle_system_state[0], particle_system_state[1]);
	std::swap(particle_buffer[0], particle_buffer[1]);
}

void GPUParticleSystem::render(VkCommandBuffer cmd)
{
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 2);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(system_globals.buffer),
		DescriptorInfo(particle_buffer[0].buffer),
		DescriptorInfo(particle_system_state[0].buffer),
		DescriptorInfo(particle_buffer[1].buffer),
		DescriptorInfo(particle_system_state[1].buffer),
		DescriptorInfo(indirect_dispatch_buffer.buffer),
		DescriptorInfo(sort_keyval_buffer[0].buffer),
		DescriptorInfo(particle_aabbs.buffer),
		DescriptorInfo(instances_buffer.buffer),
		DescriptorInfo(tlas.acceleration_structure)
	};
	vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);

	GPUParticlePushConstants pc{};
	pc.particle_size = particle_size;
	pc.particle_color = particle_color;
	vkCmdPushConstants(cmd, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
	vkCmdDrawIndirect(cmd, indirect_dispatch_buffer.buffer, offsetof(GPUParticleIndirectData, draw_cmd), 1, sizeof(GPUParticleIndirectData));

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 3);
}

void GPUParticleSystem::destroy()
{
	if (sort_context)
	{
		radix_sort_context_destroy(sort_context); 
		sort_context = nullptr;
	}
	vkDestroyAccelerationStructureKHR(ctx->device, tlas.acceleration_structure, nullptr);
	ctx->destroy_buffer(tlas.acceleration_structure_buffer);
	ctx->destroy_buffer(tlas.scratch_buffer);
	ctx->destroy_buffer(instances_buffer);
	render_pipeline->builder.destroy_resources(render_pipeline->pipeline);
	particle_emit_pipeline->builder.destroy_resources(particle_emit_pipeline->pipeline);
	particle_dispatch_size_pipeline->builder.destroy_resources(particle_dispatch_size_pipeline->pipeline);
	particle_simulate_pipeline->builder.destroy_resources(particle_simulate_pipeline->pipeline);
	particle_compact_pipeline->builder.destroy_resources(particle_compact_pipeline->pipeline);
	particle_debug_sort_pipeline->builder.destroy_resources(particle_debug_sort_pipeline->pipeline);
	ctx->destroy_buffer(system_globals);
	vkDestroyQueryPool(ctx->device, query_pool, nullptr);
	ctx->destroy_buffer(indirect_dispatch_buffer);
	ctx->destroy_buffer(sort_indirect_buffer);
	ctx->destroy_buffer(sort_internal_buffer);
	vkDestroyAccelerationStructureKHR(ctx->device, blas.acceleration_structure, nullptr);
	ctx->destroy_buffer(blas.acceleration_structure_buffer);
	ctx->destroy_buffer(blas.scratch_buffer);
	ctx->destroy_buffer(particle_aabbs);
	for (int i = 0; i < 2; ++i)
	{
		ctx->destroy_buffer(sort_keyval_buffer[i]);
		ctx->destroy_buffer(particle_buffer[i]);
		ctx->destroy_buffer(particle_system_state[i]);
	}
}

void GPUParticleSystem::draw_stats_overlay()
{
	ImGui::Begin("GPU Particle System");
	ImGui::Text("Simulation time: %f us", performance_timings.simulate_total * 1e-3f);
	ImGui::End();
}

void GPUParticleSystem::draw_ui()
{
	ImGui::SliderFloat("particle size", &particle_size, 0.001f, 1.0f);
	ImGui::SliderFloat("particle alpha", &particle_color.a, 0.01f, 1.0f);
	ImGui::ColorEdit3("particle color", glm::value_ptr(particle_color));
	ImGui::Checkbox("sort particles", &sort_particles);
}
