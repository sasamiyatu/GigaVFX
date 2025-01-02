#include "gpu_particles.h"
#include "pipeline.h"
#include "graphics_context.h"
#include "hot_reload.h"
#include "../shaders/shared.h"
#include "imgui/imgui.h"
#include "camera.h"
#include "vk_helpers.h"

constexpr VkFormat PARTICLE_RENDER_TARGET_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat LIGHT_RENDER_TARGET_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr uint32_t MIN_SLICES = 1;
constexpr uint32_t MAX_SLICES = 128;
constexpr float MAX_DELTA_TIME = 0.1f;

static uint32_t get_dispatch_size(uint32_t particle_capacity)
{
	return (particle_capacity + 63) / 64;
}

void GPUParticleSystem::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity, const Texture& shadowmap_texture, uint32_t cascade_index)
{
	assert(shadowmap_texture.width != 0);

	this->ctx = ctx;
	shader_globals = globals_buffer;
	this->particle_capacity = particle_capacity;
	this->light_buffer_size = shadowmap_texture.width;

	{ // Create image view for shadow map
		VkImageViewCreateInfo cinfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
		cinfo.image = shadowmap_texture.image;
		cinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		cinfo.format = shadowmap_texture.format;
		cinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		cinfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		cinfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		cinfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		cinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		cinfo.subresourceRange.baseArrayLayer = cascade_index;
		cinfo.subresourceRange.baseMipLevel = 0;
		cinfo.subresourceRange.layerCount = 1;
		cinfo.subresourceRange.levelCount = 1;
		
		vkCreateImageView(ctx->device, &cinfo, nullptr, &light_depth_view);
	}

	{ // Render pipeline
		GraphicsPipelineBuilder builder(ctx->device, true);
		builder
			.set_vertex_shader_filepath("gpu_particles.hlsl")
			.set_fragment_shader_filepath("gpu_particles.hlsl", "particle_fs_shadowed")
			.set_cull_mode(VK_CULL_MODE_NONE)
			.add_color_attachment(render_target_format)
			.set_depth_format(VK_FORMAT_D32_SFLOAT)
			.set_depth_test(VK_TRUE)
			.set_depth_write(VK_FALSE)
			.set_depth_compare_op(VK_COMPARE_OP_LESS)
			//.set_blend_preset(BlendPreset::ADDITIVE)
			//.set_blend_preset(BlendPreset::ALPHA)
			// "Over" operator
			.set_blend_state({
				VK_TRUE,
				VK_BLEND_FACTOR_ONE,
				VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
				VK_BLEND_OP_ADD,
				VK_BLEND_FACTOR_ONE,
				VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
				VK_BLEND_OP_ADD,
				VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
			})
			.set_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

		render_pipeline_back_to_front = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(render_pipeline_back_to_front);

		// "Under" operator
		builder.set_blend_state({
			VK_TRUE,
			VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
			VK_BLEND_FACTOR_ONE,
			VK_BLEND_OP_ADD,
			VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
			VK_BLEND_FACTOR_ONE,
			VK_BLEND_OP_ADD,
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
		});

		render_pipeline_front_to_back = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(render_pipeline_front_to_back);
	}
	 
	{ // Light render pipeline
		GraphicsPipelineBuilder builder(ctx->device, true);
		builder
			.set_vertex_shader_filepath("gpu_particles.hlsl", "vs_light")
			.set_fragment_shader_filepath("gpu_particles.hlsl", "particle_fs_light")
			.set_cull_mode(VK_CULL_MODE_NONE)
			.add_color_attachment(LIGHT_RENDER_TARGET_FORMAT)
			.set_depth_format(VK_FORMAT_D32_SFLOAT)
			.set_depth_test(VK_TRUE)
			.set_depth_write(VK_FALSE)
			.set_depth_compare_op(VK_COMPARE_OP_LESS)
			.set_blend_state({
				VK_TRUE,
				VK_BLEND_FACTOR_ONE,
				VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
				VK_BLEND_OP_ADD,
				VK_BLEND_FACTOR_ONE,
				VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
				VK_BLEND_OP_ADD,
				VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
				})
			.set_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

		render_pipeline_light = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(render_pipeline_light);
	}

	ctx->create_texture(particle_render_target, ctx->window_width, ctx->window_height, 1, PARTICLE_RENDER_TARGET_FORMAT, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
	ctx->create_texture(light_render_target, light_buffer_size, light_buffer_size, 1, LIGHT_RENDER_TARGET_FORMAT, VK_IMAGE_TYPE_2D, 
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

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

	{ // Indirect dispatch size write pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_write_draw");
		particle_draw_count_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_draw_count_pipeline);
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

	{ // Composite pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particle_composite.hlsl", "cs_composite_image");
		particle_composite_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_composite_pipeline);
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
			desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
			particle_system_state[i] = ctx->create_buffer(desc);
		}
	}

	{ // Indirect dispatch buffer
		BufferDesc desc{};
		desc.size = sizeof(DispatchIndirectCommand);
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		indirect_dispatch_buffer = ctx->create_buffer(desc);
	}

	{ // Indirect draw buffer
		BufferDesc desc{};
		desc.size = sizeof(DrawIndirectCommand) * MAX_SLICES;
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		indirect_draw_buffer = ctx->create_buffer(desc);
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

	{ // Sampler for sampling light buffer
		{
			VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
			info.magFilter = VK_FILTER_LINEAR;
			info.minFilter = VK_FILTER_LINEAR;
			info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			info.maxLod = VK_LOD_CLAMP_NONE;
			info.maxAnisotropy = 1;
			VK_CHECK(vkCreateSampler(ctx->device, &info, nullptr, &light_sampler));
		}
	}
}

void GPUParticleSystem::simulate(VkCommandBuffer cmd, float dt, CameraState& camera_state, glm::mat4 shadow_view, glm::mat4 shadow_projection)
{
	dt = glm::clamp(dt, 0.0f, MAX_DELTA_TIME);
	particles_to_spawn += particle_spawn_rate * dt;
	time += dt;

	glm::vec3 light_dir = glm::vec3(glm::inverse(shadow_view)[2]);
	glm::vec3 view_dir = -camera_state.forward;
	glm::vec3 half_vector;
	float dp = glm::dot(view_dir, light_dir);
	if (dp > 0.0f)
	{
		half_vector = glm::normalize(view_dir + light_dir);
		draw_order_flipped = false;
	}
	else
	{
		half_vector = normalize(-view_dir + light_dir);
		draw_order_flipped = true;
	}

	particle_sort_axis = -half_vector;
	//particle_sort_axis = -view_dir;
	//draw_order_flipped = true;

	{ // Update per frame globals
		// TODO: Use staging buffer for this
		GPUParticleSystemGlobals globals{};
		globals.particle_capacity = particle_capacity;
		globals.transform = glm::mat4(1.0f);

		//glm::mat4 shadow_proj = glm::ortho(-3.0f, 3.0f, -3.0f, 3.0f, 0.1f, 100.0f);
		glm::vec3 center = glm::vec3(0.0f);
		//glm::mat4 shadow_view = glm::lookAt(center + light_dir * 50.0f, center, glm::vec3(0.0f, 1.0f, 0.0));
		globals.light_view = shadow_view;
		globals.light_proj = shadow_projection;
		globals.light_resolution = glm::uvec2(light_buffer_size, light_buffer_size);

		void* mapped;
		vmaMapMemory(ctx->allocator, system_globals.allocation, &mapped);
		memcpy(mapped, &globals, sizeof(globals));
		vmaUnmapMemory(ctx->allocator, system_globals.allocation);
	}

	static bool first = true;
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
		DescriptorInfo(tlas.acceleration_structure),
		DescriptorInfo(indirect_draw_buffer.buffer),
		DescriptorInfo(light_sampler),
		DescriptorInfo(light_render_target.view, VK_IMAGE_LAYOUT_GENERAL),
	};

	// Likewise for push constants
	GPUParticlePushConstants push_constants{};
	push_constants.delta_time = dt;
	push_constants.particles_to_spawn = (uint32_t)particles_to_spawn;
	push_constants.particle_size = particle_size;
	push_constants.particle_color = particle_color;
	push_constants.sort_axis = particle_sort_axis;
	push_constants.blas_address = VkHelpers::get_acceleration_structure_device_address(ctx->device, blas.acceleration_structure);
	push_constants.emitter_radius = emitter_radius;
	push_constants.speed = particle_speed;
	push_constants.time = time;
	push_constants.lifetime = particle_lifetime;
	push_constants.noise_scale = noise_scale;
	push_constants.noise_time_scale = noise_time_scale;

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

	{ // Write indirect draw counts
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_draw_count_pipeline->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_draw_count_pipeline->pipeline.descriptor_update_template,
			particle_draw_count_pipeline->pipeline.layout, 0, descriptor_info);
		GPUParticlePushConstants pc{};
		pc.num_slices = num_slices;
		vkCmdPushConstants(cmd, particle_draw_count_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
		vkCmdDispatch(cmd, (num_slices + 63) / 64, 1, 1);

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

	{ // Sort particles
		if (sort_particles)
		{
			radix_sort_vk_sort_indirect_info sort_info{};
			sort_info.key_bits = 32;
			sort_info.count = { particle_system_state[1].buffer, 0, VK_WHOLE_SIZE};
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

#if 0

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
#endif

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

void GPUParticleSystem::render(VkCommandBuffer cmd, const Texture& render_target, const Texture& depth_target)
{
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 2);

	{ // Transition render targets
		VkImageMemoryBarrier2 barriers[2] = {
			VkHelpers::image_memory_barrier2
			(
				VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				0,
				VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
				VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_GENERAL,
				particle_render_target.image,
				VK_IMAGE_ASPECT_COLOR_BIT
			),
			VkHelpers::image_memory_barrier2
			(
				VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				0,
				VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
				VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_GENERAL,
				light_render_target.image,
				VK_IMAGE_ASPECT_COLOR_BIT
			),
		};

		VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dep_info.imageMemoryBarrierCount = 2;
		dep_info.pImageMemoryBarriers = barriers;
		vkCmdPipelineBarrier2(cmd, &dep_info);
	}

	auto render_slice_light = [&](uint32_t slice)
		{
			char marker_name[64];
			sprintf(marker_name, "Light slice %u", slice);

			VkDebugUtilsLabelEXT label_info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
			label_info.pLabelName = marker_name;
			label_info.color[0] = 0.0f;
			label_info.color[1] = 0.0f;
			label_info.color[2] = 1.0f;
			label_info.color[3] = 1.0f;
			vkCmdBeginDebugUtilsLabelEXT(cmd, &label_info);

			VkRenderingAttachmentInfo color_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			color_info.imageView = light_render_target.view;
			color_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			color_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; 
			color_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

			VkRenderingAttachmentInfo depth_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			depth_info.imageView = light_depth_view;
			depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			depth_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			depth_info.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

			VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
			rendering_info.renderArea = { {0, 0}, {light_buffer_size, light_buffer_size} };
			rendering_info.layerCount = 1;
			rendering_info.viewMask = 0;
			rendering_info.colorAttachmentCount = 1;
			rendering_info.pColorAttachments = &color_info;
			rendering_info.pDepthAttachment = &depth_info;

			vkCmdBeginRendering(cmd, &rendering_info);

			VkRect2D scissor = { {0, 0}, {light_buffer_size, light_buffer_size} };
			vkCmdSetScissor(cmd, 0, 1, &scissor);
			VkViewport viewport = { 0.0f, (float)light_buffer_size, (float)light_buffer_size, -(float)light_buffer_size, 0.0f, 1.0f };
			//VkViewport viewport = { 0.0f, 0.0f, (float)light_buffer_size, (float)light_buffer_size, 0.0f, 1.0f };
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline_light->pipeline.pipeline);
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
			vkCmdPushDescriptorSetWithTemplateKHR(cmd, 
				render_pipeline_light->pipeline.descriptor_update_template, 
				render_pipeline_light->pipeline.layout, 0, descriptor_info);

			glm::vec4 color = glm::vec4(color_attenuation * shadow_alpha, 1.0f);
			//glm::vec4 color = glm::vec4(glm::vec3(1.0f), shadow_alpha);
			GPUParticlePushConstants pc{};
			pc.particle_size = particle_size;
			pc.particle_color = color;
			vkCmdPushConstants(cmd, render_pipeline_light->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
			VkDeviceSize offset = sizeof(DrawIndirectCommand) * slice;
			vkCmdDrawIndirect(cmd, indirect_draw_buffer.buffer, offset, 1, sizeof(DrawIndirectCommand));

			vkCmdEndRendering(cmd);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		};

	auto render_slice_view = [&](uint32_t slice, bool flipped)
		{
			char marker_name[64];
			sprintf(marker_name, "View slice %u", slice);

			VkDebugUtilsLabelEXT label_info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
			label_info.pLabelName = marker_name;
			label_info.color[0] = 0.0f;
			label_info.color[1] = 1.0f;
			label_info.color[2] = 0.0f;
			label_info.color[3] = 1.0f;
			vkCmdBeginDebugUtilsLabelEXT(cmd, &label_info);

			VkRenderingAttachmentInfo color_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			color_info.imageView = particle_render_target.view;
			color_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			color_info.loadOp = slice == 0 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
			color_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			color_info.clearValue.color = { 0.0f, 0.0f, 0.0f, 0.0f };

			VkRenderingAttachmentInfo depth_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			depth_info.imageView = depth_target.view;
			depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			depth_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			depth_info.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

			VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
			rendering_info.renderArea = { {0, 0}, {(uint32_t)ctx->window_width, (uint32_t)ctx->window_height} };
			rendering_info.layerCount = 1;
			rendering_info.viewMask = 0;
			rendering_info.colorAttachmentCount = 1;
			rendering_info.pColorAttachments = &color_info;
			rendering_info.pDepthAttachment = &depth_info;

			vkCmdBeginRendering(cmd, &rendering_info);

			VkRect2D scissor = { {0, 0}, {(uint32_t)ctx->window_width, (uint32_t)ctx->window_height} };
			vkCmdSetScissor(cmd, 0, 1, &scissor);
			VkViewport viewport = { 0.0f, (float)ctx->window_height, (float)ctx->window_width, -(float)ctx->window_height, 0.0f, 1.0f };
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			const GraphicsPipelineAsset* render_pipeline = flipped ? render_pipeline_back_to_front : render_pipeline_front_to_back;
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
				DescriptorInfo(tlas.acceleration_structure),
				DescriptorInfo(indirect_draw_buffer.buffer),
				DescriptorInfo(light_sampler),
				DescriptorInfo(light_render_target.view, VK_IMAGE_LAYOUT_GENERAL)
			};
			vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);

			GPUParticlePushConstants pc{};
			pc.particle_size = particle_size;
			pc.particle_color = particle_color;
			vkCmdPushConstants(cmd, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

			VkDeviceSize offset = sizeof(DrawIndirectCommand) * slice;
			vkCmdDrawIndirect(cmd, indirect_draw_buffer.buffer, offset, 1, sizeof(DrawIndirectCommand));

			vkCmdEndRendering(cmd);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		};

	{ // Clear light buffer
		glm::vec3 light_color = glm::vec3(1.0f);
		VkClearColorValue clear{};
		clear.float32[0] = 1.0f - light_color.r;
		clear.float32[1] = 1.0f - light_color.g;
		clear.float32[2] = 1.0f - light_color.b;
		clear.float32[3] = 0.0f;
		//clear.float32[0] = light_color.r;
		//clear.float32[1] = light_color.g;
		//clear.float32[2] = light_color.b;
		//clear.float32[3] = 0.0f;

		VkImageSubresourceRange range{};
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;
		vkCmdClearColorImage(cmd, light_render_target.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
	}

	for (uint32_t i = 0; i < slices_to_display; ++i)
	{
		render_slice_light(i);
		VkHelpers::full_barrier(cmd);
		render_slice_view(i, draw_order_flipped);
		VkHelpers::full_barrier(cmd);
	}

	{ // Composite
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_composite_pipeline->pipeline.pipeline);
		DescriptorInfo descriptor_info[] = {
			DescriptorInfo(particle_render_target.view, VK_IMAGE_LAYOUT_GENERAL),
			DescriptorInfo(render_target.view, VK_IMAGE_LAYOUT_GENERAL),
		};

		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_composite_pipeline->pipeline.descriptor_update_template,
			particle_composite_pipeline->pipeline.layout, 0, descriptor_info);

		vkCmdDispatch(cmd, (ctx->window_width + 7) / 8, (ctx->window_height + 7) / 8, 1);

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

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 3);
}

void GPUParticleSystem::destroy()
{
	if (sort_context)
	{
		radix_sort_context_destroy(sort_context); 
		sort_context = nullptr;
	}
	vkDestroyImageView(ctx->device, light_depth_view, nullptr);
	vkDestroyAccelerationStructureKHR(ctx->device, tlas.acceleration_structure, nullptr);
	ctx->destroy_buffer(tlas.acceleration_structure_buffer);
	ctx->destroy_buffer(tlas.scratch_buffer);
	ctx->destroy_buffer(instances_buffer);
	particle_render_target.destroy(ctx->device, ctx->allocator);
	light_render_target.destroy(ctx->device, ctx->allocator);
	vkDestroySampler(ctx->device, light_sampler, nullptr);
	render_pipeline_back_to_front->builder.destroy_resources(render_pipeline_back_to_front->pipeline);
	render_pipeline_front_to_back->builder.destroy_resources(render_pipeline_front_to_back->pipeline);
	render_pipeline_light->builder.destroy_resources(render_pipeline_light->pipeline);
	particle_emit_pipeline->builder.destroy_resources(particle_emit_pipeline->pipeline);
	particle_dispatch_size_pipeline->builder.destroy_resources(particle_dispatch_size_pipeline->pipeline);
	particle_draw_count_pipeline->builder.destroy_resources(particle_draw_count_pipeline->pipeline);
	particle_simulate_pipeline->builder.destroy_resources(particle_simulate_pipeline->pipeline);
	particle_compact_pipeline->builder.destroy_resources(particle_compact_pipeline->pipeline);
	particle_debug_sort_pipeline->builder.destroy_resources(particle_debug_sort_pipeline->pipeline);
	particle_composite_pipeline->builder.destroy_resources(particle_composite_pipeline->pipeline);
	ctx->destroy_buffer(system_globals);
	vkDestroyQueryPool(ctx->device, query_pool, nullptr);
	ctx->destroy_buffer(indirect_dispatch_buffer);
	ctx->destroy_buffer(indirect_draw_buffer);
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
	if (ImGui::InputFloat("emission rate", &particle_spawn_rate, 1.0f, 100.0f))
	{
		particle_spawn_rate = std::clamp(particle_spawn_rate, 0.0f, 10000000.0f);
	}
	ImGui::SliderFloat("emitter radius", &emitter_radius, 0.0f, 2.0f);
	ImGui::SliderFloat("particle speed", &particle_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("particle size", &particle_size, 0.001f, 1.0f);
	ImGui::SliderFloat("particle lifetime", &particle_lifetime, 0.0f, 20.0f);
	ImGui::SliderFloat("particle alpha", &particle_color.a, 0.01f, 1.0f);
	ImGui::ColorEdit3("particle color", glm::value_ptr(particle_color));
	ImGui::SliderFloat("noise scale", &noise_scale, 0.0f, 10.0f);
	ImGui::SliderFloat("noise time scale", &noise_time_scale, 0.0f, 10.0f);
	ImGui::Checkbox("sort particles", &sort_particles);

	if (ImGui::SliderScalar("number of slices", ImGuiDataType_U32, &num_slices, &MIN_SLICES, &MAX_SLICES))
	{
		slices_to_display = std::min(slices_to_display, (int)num_slices);
	}
	if (ImGui::InputInt("slices to display", &slices_to_display, 1, 10))
	{
		slices_to_display = std::clamp(slices_to_display, 0, (int)num_slices);
	}
	ImGui::Checkbox("display single slice", &display_single_slice);
	ImGui::SliderFloat("shadow alpha", &shadow_alpha, 0.0f, 1.0f);
	ImGui::ColorEdit3("color attenuation", glm::value_ptr(color_attenuation));
}
