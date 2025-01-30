#include "gpu_particles.h"
#include "pipeline.h"
#include "graphics_context.h"
#include "hot_reload.h"
#include "../shaders/shared.h"
#include "imgui/imgui.h"
#include "camera.h"
#include "vk_helpers.h"
#include "sdf.h"
#include "colors.h"

constexpr VkFormat PARTICLE_RENDER_TARGET_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat LIGHT_RENDER_TARGET_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr uint32_t MIN_SLICES = 1;
constexpr uint32_t MAX_SLICES = 128;
constexpr float MAX_DELTA_TIME = 0.1f;

static_assert(sizeof(GPUParticlePushConstants) <= 128);

static ComputePipelineAsset* create_pipeline(Context* ctx, const char* shader_src, const char* entry_point)
{
	ComputePipelineBuilder builder(ctx->device, true);
	builder.set_shader_filepath(shader_src, entry_point);
	ComputePipelineAsset* pipeline = new ComputePipelineAsset(builder);
	AssetCatalog::register_asset(pipeline);

	return pipeline;
}

static ComputePipelineAsset* create_pipeline(Context* ctx, const ShaderSource& shader_source)
{
	ComputePipelineBuilder builder(ctx->device, true);
	builder.set_shader_source(shader_source);
	ComputePipelineAsset* pipeline = new ComputePipelineAsset(builder);
	AssetCatalog::register_asset(pipeline);

	return pipeline;
}


static void dispatch(VkCommandBuffer cmd, ComputePipelineAsset* pipeline, void* push_constants, size_t push_constant_size, void* descriptor_info,
	uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z)
{
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline.pipeline);
	vkCmdPushDescriptorSetWithTemplateKHR(cmd, pipeline->pipeline.descriptor_update_template,
		pipeline->pipeline.layout, 0, descriptor_info);

	if (push_constants)
		vkCmdPushConstants(cmd, pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size, push_constants);
	vkCmdDispatch(cmd, dispatch_x, dispatch_y, dispatch_z);
}

static void dispatch_indirect(VkCommandBuffer cmd, ComputePipelineAsset* pipeline, void* push_constants, size_t push_constant_size, void* descriptor_info,
	VkBuffer indirect_buffer, VkDeviceSize buffer_offset)
{
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline.pipeline);
	vkCmdPushDescriptorSetWithTemplateKHR(cmd, pipeline->pipeline.descriptor_update_template,
		pipeline->pipeline.layout, 0, descriptor_info);

	if (push_constants)
		vkCmdPushConstants(cmd, pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size, push_constants);
	vkCmdDispatchIndirect(cmd, indirect_buffer, buffer_offset);
}

static void compute_barrier_simple(VkCommandBuffer cmd)
{
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

static uint32_t get_dispatch_size(uint32_t particle_capacity)
{
	return (particle_capacity + 63) / 64;
}

void GPUParticleSystem::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity, 
	const Texture& shadowmap_texture, uint32_t cascade_index, const ShaderInfo& emit_shader, const ShaderInfo& update_shader, 
	bool emit_once)
{
	assert(shadowmap_texture.width != 0);

	this->ctx = ctx;
	shader_globals = globals_buffer;
	this->particle_capacity = particle_capacity;
	this->light_buffer_size = shadowmap_texture.width;
	one_time_emit = emit_once;

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
		builder.set_shader_filepath(emit_shader.shader_source_file.c_str(), emit_shader.entry_point.c_str());
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
		builder.set_shader_filepath(update_shader.shader_source_file.c_str(), update_shader.entry_point.c_str());
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
		system_globals = ctx->create_gpu_buffer(desc);
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
	VkHelpers::begin_label(cmd, "Particle system simulate", glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));

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

	{ // Update per frame globals
		// TODO: Use staging buffer for this
		GPUParticleSystemGlobals globals{};
		globals.particle_capacity = particle_capacity;
		//globals.transform = glm::mat4(1.0f);
		globals.transform = glm::translate(position);

		globals.light_view = shadow_view;
		globals.light_proj = shadow_projection;
		globals.light_resolution = glm::uvec2(light_buffer_size, light_buffer_size);

		void* mapped;
		ctx->map_buffer(system_globals, &mapped);
		memcpy(mapped, &globals, sizeof(globals));
		ctx->unmap_buffer(system_globals);
		ctx->upload_buffer(system_globals, cmd);

		VkHelpers::memory_barrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	}

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
		DescriptorInfo(system_globals),
		DescriptorInfo(particle_buffer[0].buffer),
		DescriptorInfo(particle_system_state[0].buffer),
		DescriptorInfo(particle_buffer[1].buffer),
		DescriptorInfo(particle_system_state[1].buffer),
		DescriptorInfo(indirect_dispatch_buffer.buffer),
		DescriptorInfo(sort_keyval_buffer[0].buffer),
		DescriptorInfo(particle_aabbs.buffer),
		DescriptorInfo(instances_buffer.buffer),
		DescriptorInfo(indirect_draw_buffer.buffer),
		DescriptorInfo(light_sampler),
		DescriptorInfo(light_render_target.view, VK_IMAGE_LAYOUT_GENERAL),
	};

	// Likewise for push constants
	GPUParticlePushConstants push_constants{};
	push_constants.delta_time = dt;
	push_constants.particles_to_spawn = one_time_emit ? particle_capacity :  (uint32_t)particles_to_spawn;
	push_constants.particle_size = particle_size;
	push_constants.particle_color = particle_color;
	push_constants.sort_axis = particle_sort_axis;
	push_constants.emitter_radius = emitter_radius;
	push_constants.speed = particle_speed;
	push_constants.time = time;
	push_constants.lifetime = particle_lifetime;
	push_constants.noise_scale = noise_scale;
	push_constants.noise_time_scale = noise_time_scale;
	push_constants.particle_capacity = particle_capacity;
	push_constants.smoke_dir = smoke_dir;
	push_constants.smoke_origin = smoke_origin;


	{ // Clear output state
		vkCmdFillBuffer(cmd, particle_system_state[1].buffer, 0, VK_WHOLE_SIZE, 0);
		//vkCmdFillBuffer(cmd, particle_buffer[1].buffer, 0, VK_WHOLE_SIZE, 0);
		//vkCmdFillBuffer(cmd, particle_aabbs.buffer, 0, VK_WHOLE_SIZE, 0);
		//vkCmdFillBuffer(cmd, instances_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

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

	if (!one_time_emit || first_frame)
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
			sort_info.count = { particle_system_state[1].buffer, offsetof(GPUParticleSystemState, active_particle_count), sizeof(uint32_t)};
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
	{ // Debug sort
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_debug_sort_pipeline->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_debug_sort_pipeline->pipeline.descriptor_update_template,
			particle_debug_sort_pipeline->pipeline.layout, 0, descriptor_info);
		vkCmdPushConstants(cmd, particle_debug_sort_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);
		vkCmdDispatch(cmd, 1, 1, 1);
	}
#endif

	particles_to_spawn -= std::floor(particles_to_spawn);

	// Swap buffers
	std::swap(particle_system_state[0], particle_system_state[1]);
	std::swap(particle_buffer[0], particle_buffer[1]);

	first_frame = false;

	VkHelpers::end_label(cmd);
}

void GPUParticleSystem::render(VkCommandBuffer cmd, const Texture& depth_target)
{
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

			VkHelpers::begin_label(cmd, marker_name, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

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
				DescriptorInfo(system_globals),
				DescriptorInfo(particle_buffer[0].buffer),
				DescriptorInfo(particle_system_state[0].buffer),
				DescriptorInfo(particle_buffer[1].buffer),
				DescriptorInfo(particle_system_state[1].buffer),
				DescriptorInfo(indirect_dispatch_buffer.buffer),
				DescriptorInfo(sort_keyval_buffer[0].buffer),
				DescriptorInfo(particle_aabbs.buffer),
				DescriptorInfo(instances_buffer.buffer),
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

			VkHelpers::end_label(cmd);
		};

	auto render_slice_view = [&](uint32_t slice, bool flipped)
		{
			char marker_name[64];
			sprintf(marker_name, "View slice %u", slice);

			VkHelpers::begin_label(cmd, marker_name, glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));

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
				DescriptorInfo(system_globals),
				DescriptorInfo(particle_buffer[0].buffer),
				DescriptorInfo(particle_system_state[0].buffer),
				DescriptorInfo(particle_buffer[1].buffer),
				DescriptorInfo(particle_system_state[1].buffer),
				DescriptorInfo(indirect_dispatch_buffer.buffer),
				DescriptorInfo(sort_keyval_buffer[0].buffer),
				DescriptorInfo(particle_aabbs.buffer),
				DescriptorInfo(instances_buffer.buffer),
				DescriptorInfo(indirect_draw_buffer.buffer),
				DescriptorInfo(light_sampler),
				DescriptorInfo(light_render_target.view, VK_IMAGE_LAYOUT_GENERAL)
			};
			vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);

			GPUParticlePushConstants pc{};
			pc.particle_size = particle_size;
			pc.particle_color = particle_color;
			vkCmdPushConstants(cmd, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

			VkDeviceSize offset = sizeof(DrawIndirectCommand) * slice;
			vkCmdDrawIndirect(cmd, indirect_draw_buffer.buffer, offset, 1, sizeof(DrawIndirectCommand));

			vkCmdEndRendering(cmd);

			VkHelpers::end_label(cmd);
		};

	{ // Clear light buffer
		glm::vec3 light_color = glm::vec3(1.0f);
		VkClearColorValue clear{};
		clear.float32[0] = 1.0f - light_color.r;
		clear.float32[1] = 1.0f - light_color.g;
		clear.float32[2] = 1.0f - light_color.b;
		clear.float32[3] = 0.0f;

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
		VkHelpers::fragment_barrier_simple(cmd);
		render_slice_view(i, draw_order_flipped);
		VkHelpers::fragment_barrier_simple(cmd);
	}
}

void GPUParticleSystem::composite(VkCommandBuffer cmd, const Texture& render_target)
{
	VkHelpers::begin_label(cmd, "Half angle slice composite", glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
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

	VkHelpers::end_label(cmd);
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

void GPUParticleSystem::draw_config_ui()
{
	if (ImGui::InputFloat("emission rate", &particle_spawn_rate, 100.0f, 10000.0f))
	{
		particle_spawn_rate = std::clamp(particle_spawn_rate, 0.0f, 10000000.0f);
	}
	ImGui::SliderFloat("emitter radius", &emitter_radius, 0.0f, 2.0f);
	ImGui::SliderFloat("particle speed", &particle_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("particle size", &particle_size, 0.001f, 1.0f);
	ImGui::SliderFloat("particle lifetime", &particle_lifetime, 0.0f, 20.0f);
	ImGui::SliderFloat("particle alpha", &particle_color.a, 0.01f, 1.0f);
	//ImGui::ColorEdit3("particle color", glm::value_ptr(particle_color));
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

void GPUSurfaceFlowSystem::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity, const ShaderInfo& emit_shader, const ShaderInfo& update_shader, const SDF* sdf, bool emit_once)
{
	this->ctx = ctx;
	shader_globals = globals_buffer;
	this->particle_capacity = particle_capacity;
	one_time_emit = emit_once;
	particle_size = sdf->grid_spacing;

	{ // Render pipeline
		GraphicsPipelineBuilder builder(ctx->device, true);
		builder
			.set_vertex_shader_filepath("surface_flow.hlsl")
			.set_fragment_shader_filepath("surface_flow.hlsl", "particle_fs")
			.set_cull_mode(VK_CULL_MODE_NONE)
			.add_color_attachment(render_target_format)
			.set_depth_format(VK_FORMAT_D32_SFLOAT)
			.set_depth_test(VK_TRUE)
			.set_depth_write(VK_TRUE)
			.set_depth_compare_op(VK_COMPARE_OP_LESS)
			.set_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

		render_pipeline = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(render_pipeline);
	}

	{ // Emit pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath(emit_shader.shader_source_file.c_str(), emit_shader.entry_point.c_str());
		particle_emit_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_emit_pipeline);
	}

	{ // Indirect dispatch size write pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("surface_flow.hlsl", "write_dispatch");
		particle_dispatch_size_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_dispatch_size_pipeline);
	}

	{ // Indirect dispatch size write pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("surface_flow.hlsl", "write_draw");
		particle_draw_count_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_draw_count_pipeline);
	}

	{ // Simulate pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath(update_shader.shader_source_file.c_str(), update_shader.entry_point.c_str());
		particle_simulate_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_simulate_pipeline);
	}

	{ // Compact pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("surface_flow.hlsl", "compact");
		particle_compact_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_compact_pipeline);
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

	// SDF stuff used for mesh based simulation
	{
		this->sdf = sdf;

		VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
		info.magFilter = VK_FILTER_LINEAR;
		info.minFilter = VK_FILTER_LINEAR;
		info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		info.maxLod = VK_LOD_CLAMP_NONE;
		info.maxAnisotropy = 1;
		VK_CHECK(vkCreateSampler(ctx->device, &info, nullptr, &sdf_sampler));
	}

	{ // Grid counters and grid cells for collision detection
		BufferDesc desc{};
		desc.size = sdf->dims.z * sdf->dims.y * sdf->dims.x * sizeof(uint32_t);
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		
		grid_counters = ctx->create_buffer(desc);

		desc.size = sdf->dims.z * sdf->dims.y * sdf->dims.x * max_particles_in_cell * sizeof(uint32_t);
		grid_cells = ctx->create_buffer(desc);
	}
}

void GPUSurfaceFlowSystem::simulate(VkCommandBuffer cmd, float dt)
{
	VkHelpers::begin_label(cmd, "Surface Flow simulate", glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));

	dt = glm::clamp(dt, 0.0f, MAX_DELTA_TIME);
	particles_to_spawn += particle_spawn_rate * dt;
	time += dt;

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
		DescriptorInfo(particle_buffer[0].buffer),
		DescriptorInfo(particle_system_state[0].buffer),
		DescriptorInfo(particle_buffer[1].buffer),
		DescriptorInfo(particle_system_state[1].buffer),
		DescriptorInfo(indirect_dispatch_buffer.buffer),
		DescriptorInfo(indirect_draw_buffer.buffer),
		DescriptorInfo(sdf_sampler),
		DescriptorInfo(sdf->texture.view, sdf->texture.layout),
		DescriptorInfo(grid_counters.buffer),
		DescriptorInfo(grid_cells.buffer),
	};

	// Likewise for push constants
	GPUParticlePushConstants push_constants{};
	push_constants.delta_time = dt;
	push_constants.particles_to_spawn = one_time_emit ? particle_capacity : (uint32_t)particles_to_spawn;
	push_constants.particle_size = particle_size * 0.1f;
	push_constants.particle_color = particle_color;
	push_constants.speed = particle_speed;
	push_constants.time = time;
	push_constants.sdf_grid_dims = sdf->dims;
	push_constants.sdf_grid_spacing = sdf->grid_spacing;
	push_constants.sdf_origin = sdf->grid_origin + glm::vec3(0.0f, 25.0f, 0.0f);
	push_constants.particle_capacity = particle_capacity;

	{ // Clear output state
		vkCmdFillBuffer(cmd, particle_system_state[1].buffer, 0, VK_WHOLE_SIZE, 0);
		//vkCmdFillBuffer(cmd, particle_buffer[1].buffer, 0, VK_WHOLE_SIZE, 0);
		//vkCmdFillBuffer(cmd, grid_counters.buffer, 0, VK_WHOLE_SIZE, 0);
		//vkCmdFillBuffer(cmd, grid_cells.buffer, 0, VK_WHOLE_SIZE, 0);

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

	if (!one_time_emit || first_frame)
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
		pc.num_slices = 1;
		vkCmdPushConstants(cmd, particle_draw_count_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
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

	particles_to_spawn -= std::floor(particles_to_spawn);

	// Swap buffers
	std::swap(particle_system_state[0], particle_system_state[1]);
	std::swap(particle_buffer[0], particle_buffer[1]);

	first_frame = false;

	VkHelpers::end_label(cmd);
}

void GPUSurfaceFlowSystem::render(VkCommandBuffer cmd)
{
	VkHelpers::begin_label(cmd, "Surface Flow", glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);

	// All passes use the same descriptors for now
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(particle_buffer[0].buffer),
		DescriptorInfo(particle_system_state[0].buffer),
		DescriptorInfo(particle_buffer[1].buffer),
		DescriptorInfo(particle_system_state[1].buffer),
		DescriptorInfo(indirect_dispatch_buffer.buffer),
		DescriptorInfo(indirect_draw_buffer.buffer),
		DescriptorInfo(sdf_sampler),
		DescriptorInfo(sdf->texture.view, sdf->texture.layout),
	};

	// Likewise for push constants
	GPUParticlePushConstants push_constants{};
	push_constants.particle_size = particle_size * 0.1f;
	push_constants.particle_color = particle_color;
	push_constants.speed = particle_speed;
	push_constants.time = time;
	push_constants.sdf_grid_dims = sdf->dims;
	push_constants.sdf_grid_spacing = sdf->grid_spacing;
	push_constants.sdf_origin = sdf->grid_origin + glm::vec3(0.0f, 25.0f, 0.0f);
	push_constants.particle_capacity = particle_capacity;

	vkCmdPushConstants(cmd, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_constants), &push_constants);
	vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);
	vkCmdDrawIndirect(cmd, indirect_draw_buffer.buffer, 0, 1, sizeof(DrawIndirectCommand));

	VkHelpers::end_label(cmd);
}

void GPUSurfaceFlowSystem::destroy()
{
	vkDestroySampler(ctx->device, sdf_sampler, nullptr);

	render_pipeline->builder.destroy_resources(render_pipeline->pipeline);
	particle_emit_pipeline->builder.destroy_resources(particle_emit_pipeline->pipeline);
	particle_dispatch_size_pipeline->builder.destroy_resources(particle_dispatch_size_pipeline->pipeline);
	particle_draw_count_pipeline->builder.destroy_resources(particle_draw_count_pipeline->pipeline);
	particle_simulate_pipeline->builder.destroy_resources(particle_simulate_pipeline->pipeline);
	particle_compact_pipeline->builder.destroy_resources(particle_compact_pipeline->pipeline);
	
	ctx->destroy_buffer(indirect_dispatch_buffer);
	ctx->destroy_buffer(indirect_draw_buffer);
	ctx->destroy_buffer(grid_counters);
	ctx->destroy_buffer(grid_cells);
	for (int i = 0; i < 2; ++i)
	{
		ctx->destroy_buffer(particle_buffer[i]);
		ctx->destroy_buffer(particle_system_state[i]);
	}
}

void TrailBlazerSystem::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format)
{
	this->ctx = ctx;
	shader_globals = globals_buffer;
	this->particle_capacity = particle_capacity;

	{ // Render pipeline
		ShaderSource fragment_source("trail_blazer.hlsl", "particle_fs");
		GraphicsPipelineBuilder builder(ctx->device, true);
		builder
			.set_vertex_shader_filepath("trail_blazer.hlsl")
			.set_fragment_shader_source(fragment_source)
			.set_cull_mode(VK_CULL_MODE_NONE)
			.add_color_attachment(render_target_format)
			.set_blend_preset(BlendPreset::ADDITIVE)
			.set_depth_format(VK_FORMAT_D32_SFLOAT)
			.set_depth_test(VK_TRUE)
			.set_depth_write(VK_FALSE)
			.set_depth_compare_op(VK_COMPARE_OP_LESS)
			.set_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

		render_pipeline = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(render_pipeline);
	}

	// Pipelines
	particle_emit_pipeline = create_pipeline(ctx, "trail_blazer.hlsl", "emit");
	particle_simulate_pipeline = create_pipeline(ctx, "trail_blazer.hlsl", "simulate");

	child_emit_pipeline = create_pipeline(ctx, "trail_blazer_child.hlsl", "emit");
	child_dispatch_size_pipeline = create_pipeline(ctx, "trail_blazer_child.hlsl", "write_dispatch");
	child_draw_count_pipeline = create_pipeline(ctx, "trail_blazer_child.hlsl", "write_draw");
	child_simulate_pipeline = create_pipeline(ctx, "trail_blazer_child.hlsl", "simulate");


	{ // Particles buffer
		for (int i = 0; i < 2; ++i)
		{
			size_t particle_buffer_size = particle_capacity * sizeof(GPUParticle);
			size_t child_buffer_size = child_particle_capacity * sizeof(GPUParticle);
			BufferDesc desc{};
			desc.size = particle_buffer_size;
			desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			particle_buffer[i] = ctx->create_buffer(desc);

			desc.size = child_buffer_size;
			child_particle_buffer[i] = ctx->create_buffer(desc);
		}
	}

	{ // Particle system state buffer
		for (int i = 0; i < 2; ++i)
		{
			BufferDesc desc{};
			desc.size = sizeof(GPUParticleSystemState);
			desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
				VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
			particle_system_state[i] = ctx->create_buffer(desc);
			child_particle_system_state[i] = ctx->create_buffer(desc);
 		}
	}

	{ // Child emit indirect dispatch
		BufferDesc desc{};
		desc.size = sizeof(DispatchIndirectCommand);
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		child_emit_indirect_dispatch_buffer = ctx->create_gpu_buffer(desc);
	}

	{ // indirect dispatch buffer
		BufferDesc desc{};
		desc.size = sizeof(DispatchIndirectCommand);
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | 
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		indirect_dispatch_buffer = ctx->create_buffer(desc);
		child_indirect_dispatch_buffer = ctx->create_buffer(desc);
	}

	{ //indirect draw buffer
		BufferDesc desc{};
		desc.size = sizeof(DrawIndirectCommand) * MAX_SLICES;
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | 
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		indirect_draw_buffer = ctx->create_buffer(desc);
		child_indirect_draw_buffer = ctx->create_buffer(desc);
	}

	{
		//VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
		//info.magFilter = VK_FILTER_LINEAR;
		//info.minFilter = VK_FILTER_LINEAR;
		//info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		//info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		//info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		//info.maxLod = VK_LOD_CLAMP_NONE;
		//info.maxAnisotropy = 1;
		//VK_CHECK(vkCreateSampler(ctx->device, &info, nullptr, &sdf_sampler));
	}
}

void TrailBlazerSystem::simulate(VkCommandBuffer cmd, float dt)
{
	VkHelpers::begin_label(cmd, "Trail Blazer simulate", glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));

	dt = glm::clamp(dt, 0.0f, MAX_DELTA_TIME);
	particles_to_spawn += particle_spawn_rate * dt;
	child_particles_to_spawn += child_spawn_rate * dt;
	time += dt;

	if (!particles_initialized)
	{ // Zero init buffers
		for (int i = 0; i < 2; ++i)
		{
			vkCmdFillBuffer(cmd, particle_buffer[i].buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmd, particle_system_state[i].buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmd, child_particle_buffer[i].buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmd, child_particle_system_state[i].buffer, 0, VK_WHOLE_SIZE, 0);
		}

		VkHelpers::memory_barrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

		particles_initialized = true;
	}

	{ // Parent particle system
		// All passes use the same descriptors for now
		DescriptorInfo descriptor_info[] = {
			DescriptorInfo(shader_globals),
			DescriptorInfo(particle_buffer[0].buffer),
			DescriptorInfo(particle_system_state[0].buffer),
			DescriptorInfo(particle_buffer[1].buffer),
			DescriptorInfo(particle_system_state[1].buffer),
			DescriptorInfo(indirect_dispatch_buffer.buffer),
			DescriptorInfo(indirect_draw_buffer.buffer),
			DescriptorInfo(child_emit_indirect_dispatch_buffer),
			//DescriptorInfo(sdf->texture.view, sdf->texture.layout),
			//DescriptorInfo(sdf_sampler)
		};

		// Likewise for push constants
		TrailBlazerPushConstants push_constants{};
		push_constants.delta_time = dt;
		push_constants.particles_to_spawn = (uint32_t)particles_to_spawn;
		push_constants.particle_capacity = particle_capacity;
		//push_constants.sdf_dims = sdf->dims;
		//push_constants.sdf_origin = sdf->grid_origin;
		//push_constants.sdf_spacing = sdf->grid_spacing;

		DescriptorInfo child_descriptor_info[] = {
			DescriptorInfo(shader_globals),
			DescriptorInfo(child_particle_buffer[0].buffer),
			DescriptorInfo(child_particle_system_state[0].buffer),
			DescriptorInfo(child_particle_buffer[1].buffer),
			DescriptorInfo(child_particle_system_state[1].buffer),
			DescriptorInfo(child_indirect_dispatch_buffer.buffer),
			DescriptorInfo(child_indirect_draw_buffer.buffer),
			DescriptorInfo(particle_system_state[0].buffer),
			DescriptorInfo(particle_buffer[0].buffer),
		};

		TrailBlazerPushConstants child_push_constants{};

		child_push_constants.delta_time = dt;
		child_push_constants.particles_to_spawn = (uint32_t)particles_to_spawn;
		child_push_constants.particle_capacity = child_particle_capacity;

		{ // Clear output state
			VkHelpers::begin_label(cmd, "Clear buffers", Colors::BEIGE);
			vkCmdFillBuffer(cmd, particle_system_state[1].buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmd, child_particle_system_state[1].buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmd, indirect_draw_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmd, child_indirect_draw_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

			{
				DispatchIndirectCommand dispatch{};
				dispatch.x = 0;
				dispatch.y = (uint32_t)child_particles_to_spawn;
				dispatch.z = 1;

				void* mapped = nullptr;
				ctx->map_buffer(child_emit_indirect_dispatch_buffer, &mapped);
				memcpy(mapped, &dispatch, sizeof(dispatch));
				ctx->unmap_buffer(child_emit_indirect_dispatch_buffer);
				ctx->upload_buffer(child_emit_indirect_dispatch_buffer, cmd, offsetof(DispatchIndirectCommand, y));
			}

			{
				VkBufferCopy copy{};
				copy.dstOffset = 0;
				copy.srcOffset = 0;
				copy.size = sizeof(uint32_t);
				vkCmdCopyBuffer(cmd, indirect_dispatch_buffer.buffer, child_emit_indirect_dispatch_buffer, 1, &copy);
			}

			VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

			VkHelpers::end_label(cmd);
		}

		{ // Emit particles
			VkHelpers::begin_label(cmd, "Emit", Colors::CYAN);
			dispatch(cmd, particle_emit_pipeline, &push_constants, sizeof(push_constants), descriptor_info, get_dispatch_size(push_constants.particles_to_spawn), 1, 1);
			VkHelpers::end_label(cmd);

			VkHelpers::begin_label(cmd, "Emit child", Colors::CYAN);
			dispatch_indirect(cmd, child_emit_pipeline, &child_push_constants, sizeof(child_push_constants), child_descriptor_info, child_emit_indirect_dispatch_buffer, 0);
			VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
				VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
			VkHelpers::end_label(cmd);
		}

		{ // Simulate particles
			VkHelpers::begin_label(cmd, "Simulate parent", glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
			dispatch_indirect(cmd, particle_simulate_pipeline, &push_constants, sizeof(push_constants), descriptor_info, indirect_dispatch_buffer.buffer, 0);
			VkHelpers::end_label(cmd);

			VkHelpers::begin_label(cmd, "Simulate child", Colors::LIME);
			dispatch_indirect(cmd, child_simulate_pipeline, &child_push_constants, sizeof(child_push_constants), child_descriptor_info, child_indirect_dispatch_buffer.buffer, 0);
			compute_barrier_simple(cmd);
			VkHelpers::end_label(cmd);
		}
	}	

	particles_to_spawn -= std::floor(particles_to_spawn);
	child_particles_to_spawn -= std::floor(child_particles_to_spawn);

	// Swap buffers
	std::swap(particle_system_state[0], particle_system_state[1]);
	std::swap(particle_buffer[0], particle_buffer[1]);
	std::swap(child_particle_system_state[0], child_particle_system_state[1]);
	std::swap(child_particle_buffer[0], child_particle_buffer[1]);

	first_frame = false;

	VkHelpers::end_label(cmd);
}

void TrailBlazerSystem::render(VkCommandBuffer cmd)
{
	VkHelpers::begin_label(cmd, "Trail Blazer render", glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);

	// All passes use the same descriptors for now
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(particle_buffer[0].buffer),
		DescriptorInfo(particle_system_state[0].buffer),
		DescriptorInfo(particle_buffer[1].buffer),
		DescriptorInfo(particle_system_state[1].buffer),
		DescriptorInfo(indirect_dispatch_buffer.buffer),
		DescriptorInfo(indirect_draw_buffer.buffer),
	};

	// Likewise for push constants
	GPUParticlePushConstants push_constants{};
	push_constants.particle_size = particle_size * 0.1f;
	push_constants.particle_color = particle_color;
	push_constants.speed = particle_speed;
	push_constants.time = time;
	push_constants.particle_capacity = particle_capacity;

	//vkCmdPushConstants(cmd, render_pipeline->pipeline.layout, 0 /*VK_SHADER_STAGE_VERTEX_BIT  | VK_SHADER_STAGE_FRAGMENT_BIT*/, 0, sizeof(push_constants), &push_constants);
	vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);
	vkCmdDrawIndirect(cmd, indirect_draw_buffer.buffer, 0, 1, sizeof(DrawIndirectCommand));

	{ // Draw child particles
		DescriptorInfo descriptor_info[] = {
			DescriptorInfo(shader_globals),
			DescriptorInfo(child_particle_buffer[0].buffer),
			DescriptorInfo(child_particle_system_state[0].buffer),
			DescriptorInfo(child_particle_buffer[1].buffer),
			DescriptorInfo(child_particle_system_state[1].buffer),
			DescriptorInfo(child_indirect_dispatch_buffer.buffer),
			DescriptorInfo(child_indirect_draw_buffer.buffer),
		};
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);
		vkCmdDrawIndirect(cmd, child_indirect_draw_buffer.buffer, 0, 1, sizeof(DrawIndirectCommand));
	}

	VkHelpers::end_label(cmd);
}

void TrailBlazerSystem::destroy()
{
	render_pipeline->builder.destroy_resources(render_pipeline->pipeline);
	particle_emit_pipeline->builder.destroy_resources(particle_emit_pipeline->pipeline);
	particle_simulate_pipeline->builder.destroy_resources(particle_simulate_pipeline->pipeline);

	child_emit_pipeline->builder.destroy_resources(child_emit_pipeline->pipeline);
	child_dispatch_size_pipeline->builder.destroy_resources(child_dispatch_size_pipeline->pipeline);
	child_draw_count_pipeline->builder.destroy_resources(child_draw_count_pipeline->pipeline);
	child_simulate_pipeline->builder.destroy_resources(child_simulate_pipeline->pipeline);

	//vkDestroySampler(ctx->device, sdf_sampler, nullptr);
	ctx->destroy_buffer(indirect_dispatch_buffer);
	ctx->destroy_buffer(indirect_draw_buffer);
	ctx->destroy_buffer(child_indirect_dispatch_buffer);
	ctx->destroy_buffer(child_indirect_draw_buffer);
	ctx->destroy_buffer(child_emit_indirect_dispatch_buffer);
	for (int i = 0; i < 2; ++i)
	{
		ctx->destroy_buffer(particle_buffer[i]);
		ctx->destroy_buffer(particle_system_state[i]);
		ctx->destroy_buffer(child_particle_buffer[i]);
		ctx->destroy_buffer(child_particle_system_state[i]);
	}
}

void TrailBlazerSystem::draw_config_ui()
{
	if (ImGui::InputFloat("parent emission rate", &particle_spawn_rate, 1.0f, 100.0f))
	{
		particle_spawn_rate = std::clamp(particle_spawn_rate, 0.0f, 10000000.0f);
	}
	if (ImGui::InputFloat("child emission rate", &child_spawn_rate, 100.0f, 10000.0f))
	{
		child_spawn_rate = std::clamp(child_spawn_rate, 0.0f, 10000000.0f);
	}
}

const char* TrailBlazerSystem::get_display_name()
{
	return "Trail Blazer";
}

void ParticleSystemSimple::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, const Config& cfg)
{
	this->ctx = ctx;
	shader_globals = globals_buffer;
	config = cfg;
	emit_indirect_dispatch_handled_externally = cfg.emit_indirect_dispatch_handled_externally;

	constexpr uint32_t default_descriptor_count = 5;
	descriptors.resize(default_descriptor_count + cfg.additional_descriptors.size());
	for (uint32_t i = 0; i < cfg.additional_descriptors.size(); ++i)
	{
		descriptors[default_descriptor_count + i] = cfg.additional_descriptors[i];
	}

	{ // Render pipeline
		ShaderSource vertex_source("particle_render.hlsl", "vs_main");
		ShaderSource fragment_source("particle_render.hlsl", "fs_main");
		fragment_source.add_include(cfg.emit_and_simulate_file, true);
		GraphicsPipelineBuilder builder(ctx->device, true);
		builder
			.set_vertex_shader_source(vertex_source)
			.set_fragment_shader_source(fragment_source)
			.set_cull_mode(VK_CULL_MODE_NONE)
			.add_color_attachment(render_target_format)
			//.set_blend_preset(BlendPreset::ADDITIVE)
			.set_blend_state(
				VkPipelineColorBlendAttachmentState{
					VK_TRUE,
					VK_BLEND_FACTOR_ONE,
					VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
					VK_BLEND_OP_ADD,
					VK_BLEND_FACTOR_ONE,
					VK_BLEND_FACTOR_ZERO,
					VK_BLEND_OP_ADD,
					VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
				}
			)
			.set_depth_format(VK_FORMAT_D32_SFLOAT)
			.set_depth_test(VK_TRUE)
			.set_depth_write(VK_FALSE)
			.set_depth_compare_op(VK_COMPARE_OP_LESS)
			.set_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

		render_pipeline = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(render_pipeline);
	}

	// Pipelines
	ShaderSource emit_source("particle_template.hlsl", "emit");
	emit_source.add_include(cfg.emit_and_simulate_file, true);
	particle_emit_pipeline = create_pipeline(ctx, emit_source);

	ShaderSource simulate_source("particle_template.hlsl", "simulate");
	simulate_source.add_include(cfg.emit_and_simulate_file, true);
	particle_simulate_pipeline = create_pipeline(ctx, simulate_source);

	{ // Particles buffer
		for (int i = 0; i < 2; ++i)
		{
			size_t particle_buffer_size = cfg.particle_capacity * sizeof(GPUParticle);
			BufferDesc desc{};
			desc.size = particle_buffer_size;
			desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			particle_buffer[i] = ctx->create_buffer(desc);
		}
	}

	{ // Emit indirect dispatch buffer
		BufferDesc desc{};
		desc.size = sizeof(DispatchIndirectCommand);
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		emit_indirect_dispatch_buffer = ctx->create_buffer(desc);
	}
}

void ParticleSystemSimple::pre_update(VkCommandBuffer cmd, float dt, const GPUBuffer& curr_state, const GPUBuffer& next_state, uint32_t system_index)
{
	dt = glm::clamp(dt, 0.0f, MAX_DELTA_TIME);
	particles_to_spawn += config.spawn_rate * dt;
	time += dt;

	if (!particles_initialized)
	{ // Zero init buffers
		for (int i = 0; i < 2; ++i)
		{
			vkCmdFillBuffer(cmd, particle_buffer[i].buffer, 0, VK_WHOLE_SIZE, 0);
		}

		vkCmdFillBuffer(cmd, emit_indirect_dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

		VkHelpers::memory_barrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

		particles_initialized = true;
	}

	{
		// set descriptors

		// All passes use the same descriptors for now
		DescriptorInfo descriptor_info[] = {
			DescriptorInfo(shader_globals),
			DescriptorInfo(particle_buffer[0].buffer),
			DescriptorInfo(curr_state),
			DescriptorInfo(particle_buffer[1].buffer),
			DescriptorInfo(next_state),
		};

		memcpy(descriptors.data(), descriptor_info, sizeof(descriptor_info));
	}

	{
		// Likewise for push constants
		push_constants.particles_to_spawn = (uint32_t)particles_to_spawn;
		push_constants.particle_capacity = config.particle_capacity;
		push_constants.delta_time = dt;
		push_constants.system_index = system_index;
		push_constants.externally_dispatched = emit_indirect_dispatch_handled_externally;

		{ // Clear output state
			VkHelpers::begin_label(cmd, "Clear buffers", Colors::BEIGE);

			// TODO: Don't use indirect emit when its not handled externally
			if (!emit_indirect_dispatch_handled_externally)
			{
				{
					uint32_t emit_count = push_constants.particles_to_spawn;
					GPUParticleSystemState state{};
					state.particles_to_emit = emit_count;
					void* mapped;
					ctx->map_buffer(curr_state, &mapped);
					memcpy((GPUParticleSystemState*)mapped + system_index, &state, sizeof(state));
					ctx->unmap_buffer(curr_state);
					ctx->upload_buffer(curr_state, cmd, 
						sizeof(GPUParticleSystemState) * system_index + offsetof(state, particles_to_emit), 
						sizeof(state.particles_to_emit));
				}
			}
			VkHelpers::end_label(cmd);
		}
	}
}

void ParticleSystemSimple::emit(VkCommandBuffer cmd, float dt)
{
	// Likewise for push constants
	VkHelpers::begin_label(cmd, "Emit", Colors::CYAN);
	//dispatch(cmd, particle_emit_pipeline, &push_constants, sizeof(push_constants), descriptor_info, get_dispatch_size(push_constants.particles_to_spawn), 1, 1);
	if (emit_indirect_dispatch_handled_externally)
		dispatch_indirect(cmd, particle_emit_pipeline, &push_constants, sizeof(push_constants), descriptors.data(), emit_indirect_dispatch_buffer.buffer, 0);
	else
		dispatch(cmd, particle_emit_pipeline, &push_constants, sizeof(push_constants), descriptors.data(), get_dispatch_size(push_constants.particles_to_spawn), 1, 1);
	//VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
	//	VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
	VkHelpers::end_label(cmd);
}

void ParticleSystemSimple::update(VkCommandBuffer cmd, float dt, VkBuffer indirect_dispatch_buffer, uint32_t buffer_offset)
{
	VkHelpers::begin_label(cmd, "Simulate", glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
	dispatch_indirect(cmd, particle_simulate_pipeline, &push_constants, sizeof(push_constants), descriptors.data(), indirect_dispatch_buffer, buffer_offset);
	VkHelpers::end_label(cmd);
}

void ParticleSystemSimple::post_update(VkCommandBuffer cmd, float dt)
{
	particles_to_spawn -= std::floor(particles_to_spawn);

	// Swap buffers
	std::swap(particle_buffer[0], particle_buffer[1]);
}

void ParticleSystemSimple::render(VkCommandBuffer cmd, VkBuffer indirect_draw_buffer, size_t offset)
{
	VkHelpers::begin_label(cmd, "Particle template render", glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);

	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(particle_buffer[0].buffer),
	};

	vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);
	vkCmdDrawIndirect(cmd, indirect_draw_buffer, offset, 1, sizeof(DrawIndirectCommand));

	VkHelpers::end_label(cmd);
}

void ParticleSystemSimple::destroy()
{
	render_pipeline->builder.destroy_resources(render_pipeline->pipeline);
	particle_emit_pipeline->builder.destroy_resources(particle_emit_pipeline->pipeline);
	particle_simulate_pipeline->builder.destroy_resources(particle_simulate_pipeline->pipeline);

	ctx->destroy_buffer(emit_indirect_dispatch_buffer);
	for (int i = 0; i < 2; ++i)
	{
		ctx->destroy_buffer(particle_buffer[i]);
	}
}

void ParticleSystemSimple::draw_config_ui()
{
	if (ImGui::InputFloat("emission rate", &config.spawn_rate, 1.0f, 100.0f))
	{
		config.spawn_rate = std::clamp(config.spawn_rate, 0.0f, 10000000.0f);
	}
}

const char* ParticleSystemSimple::get_display_name()
{
	return config.name.c_str();
}

void ParticleManagerSimple::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format)
{
	this->ctx = ctx;
	this->globals_buffer = globals_buffer;
	this->render_target_format = render_target_format;

	write_indirect_dispatch = create_pipeline(ctx, "particle_indirect_dispatch.hlsl", "write_dispatch");
	write_indirect_draw = create_pipeline(ctx, "particle_indirect_draw.hlsl", "write_draw");

	{
		BufferDesc desc{};
		desc.size = sizeof(GPUParticleSystemState) * MAX_SYSTEMS;
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		for (int i = 0; i < 2; ++i)
			system_states_buffer[i] = ctx->create_gpu_buffer(desc);

		desc.size = sizeof(DispatchIndirectCommand) * MAX_SYSTEMS;
		desc.usage_flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		indirect_dispatch_buffer = ctx->create_buffer(desc);

		desc.size = sizeof(DrawIndirectCommand) * MAX_SYSTEMS;
		indirect_draw_buffer = ctx->create_buffer(desc);
	}
}

ParticleSystemSimple* ParticleManagerSimple::add_system(const ParticleSystemSimple::Config& cfg)
{
	ParticleSystemSimple* system = new ParticleSystemSimple();
	system->init(ctx, globals_buffer, render_target_format, cfg);
	systems.push_back(system);

	return system;
}

void ParticleManagerSimple::update_systems(VkCommandBuffer cmd, float dt)
{
	if (first_frame)
	{
		vkCmdFillBuffer(cmd, system_states_buffer[0], 0, VK_WHOLE_SIZE, 0);

		first_frame = false;
	}

	// Clear output state
	vkCmdFillBuffer(cmd, system_states_buffer[1], 0, VK_WHOLE_SIZE, 0);
	vkCmdFillBuffer(cmd, indirect_dispatch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);
	vkCmdFillBuffer(cmd, indirect_draw_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

	VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
		VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

	VkHelpers::begin_label(cmd, "Particle Manager pre update", Colors::APRICOT);
	for (size_t i = 0; i < systems.size(); ++i) systems[i]->pre_update(cmd, dt, system_states_buffer[0], system_states_buffer[1], (uint32_t)i);

	VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
		VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
	VkHelpers::end_label(cmd);

	VkHelpers::begin_label(cmd, "Particle Manager emit", Colors::CYAN);
	for (ParticleSystemSimple* system : systems) system->emit(cmd, dt);

	VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
	VkHelpers::end_label(cmd);

	{
		VkHelpers::begin_label(cmd, "Particle Manager write indirect dispatch", Colors::MAGENTA);
		DescriptorInfo descriptor_info[] = {
			DescriptorInfo(system_states_buffer[0]),
			DescriptorInfo(indirect_dispatch_buffer.buffer),
		};

		uint32_t system_count = (uint32_t)systems.size();	

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, write_indirect_dispatch->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, write_indirect_dispatch->pipeline.descriptor_update_template,
			write_indirect_dispatch->pipeline.layout, 0, descriptor_info);
		vkCmdPushConstants(cmd, write_indirect_dispatch->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &system_count);
		vkCmdDispatch(cmd, get_dispatch_size(system_count), 1, 1);

		VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
		VkHelpers::end_label(cmd);
	}

	VkHelpers::begin_label(cmd, "Particle Manager update", Colors::LIME);
	for (size_t i = 0; i < systems.size(); ++i) systems[i]->update(cmd, dt, indirect_dispatch_buffer.buffer, i * sizeof(DispatchIndirectCommand));

	compute_barrier_simple(cmd);
	VkHelpers::end_label(cmd);

	{
		VkHelpers::begin_label(cmd, "Particle Manager write indirect draw", Colors::BLUE);
		DescriptorInfo descriptor_info[] = {
			DescriptorInfo(system_states_buffer[1]),
			DescriptorInfo(indirect_draw_buffer.buffer),
		};

		uint32_t system_count = (uint32_t)systems.size();

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, write_indirect_draw->pipeline.pipeline);
		vkCmdPushDescriptorSetWithTemplateKHR(cmd, write_indirect_draw->pipeline.descriptor_update_template,
			write_indirect_draw->pipeline.layout, 0, descriptor_info);
		vkCmdPushConstants(cmd, write_indirect_draw->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &system_count);
		vkCmdDispatch(cmd, get_dispatch_size(system_count), 1, 1);

		VkHelpers::memory_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
		VkHelpers::end_label(cmd);
	}

	for (ParticleSystemSimple* system : systems) system->post_update(cmd, dt);

	std::swap(system_states_buffer[0], system_states_buffer[1]);
}

void ParticleManagerSimple::render_systems(VkCommandBuffer cmd)
{
	VkHelpers::begin_label(cmd, "Particle Manager render", Colors::BEIGE);
	for (size_t i = 0; i < systems.size(); ++i) systems[i]->render(cmd, indirect_draw_buffer.buffer, sizeof(DrawIndirectCommand) * i);
	VkHelpers::end_label(cmd);
}

void ParticleManagerSimple::destroy()
{
	for (ParticleSystemSimple* system : systems) system->destroy();
	ctx->destroy_buffer(indirect_dispatch_buffer);
	for (int i = 0; i < 2; ++i)
		ctx->destroy_buffer(system_states_buffer[i]);
	ctx->destroy_buffer(indirect_draw_buffer);
	write_indirect_dispatch->builder.destroy_resources(write_indirect_dispatch->pipeline);
	write_indirect_draw->builder.destroy_resources(write_indirect_draw->pipeline);
}
