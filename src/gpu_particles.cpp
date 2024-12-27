#include "gpu_particles.h"
#include "pipeline.h"
#include "graphics_context.h"
#include "hot_reload.h"
#include "../shaders/shared.h"
#include "imgui/imgui.h"

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
			.set_blend_preset(BlendPreset::ADDITIVE)
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
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		indirect_dispatch_buffer = ctx->create_buffer(desc);
	}

	{ // Query pool used for perf measurement
		VkQueryPoolCreateInfo info{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
		info.queryCount = 256;
		info.queryType = VK_QUERY_TYPE_TIMESTAMP;
		VK_CHECK(vkCreateQueryPool(ctx->device, &info, nullptr, &query_pool));
	}

	{ // Radix sort context
		sort_context = radix_sort_context_create(ctx, particle_capacity);
	}
}

void GPUParticleSystem::simulate(VkCommandBuffer cmd, float dt)
{
	static bool first = true;
	particles_to_spawn += particle_spawn_rate * dt;

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
	};

	{ // Clear output state
		vkCmdFillBuffer(cmd, particle_system_state[1].buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmd, particle_buffer[1].buffer, 0, VK_WHOLE_SIZE, 0);

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
		GPUParticlePushConstants pc{};
		pc.delta_time = dt;
		pc.particles_to_spawn = (uint32_t)particles_to_spawn;

		vkCmdPushConstants(cmd, particle_emit_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
		vkCmdDispatch(cmd, get_dispatch_size(pc.particles_to_spawn), 1, 1);

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
		GPUParticlePushConstants pc{};
		pc.delta_time = dt;
		pc.particles_to_spawn = (uint32_t)particles_to_spawn;
		particles_to_spawn -= std::floor(particles_to_spawn);
		vkCmdPushConstants(cmd, particle_simulate_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
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
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);
	}

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
	};
	vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);
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
	render_pipeline->builder.destroy_resources(render_pipeline->pipeline);
	particle_emit_pipeline->builder.destroy_resources(particle_emit_pipeline->pipeline);
	particle_dispatch_size_pipeline->builder.destroy_resources(particle_dispatch_size_pipeline->pipeline);
	particle_simulate_pipeline->builder.destroy_resources(particle_simulate_pipeline->pipeline);
	particle_compact_pipeline->builder.destroy_resources(particle_compact_pipeline->pipeline);
	ctx->destroy_buffer(system_globals);
	vkDestroyQueryPool(ctx->device, query_pool, nullptr);
	ctx->destroy_buffer(indirect_dispatch_buffer);
	for (int i = 0; i < 2; ++i)
	{
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
