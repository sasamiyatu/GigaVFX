#include "gpu_particles.h"
#include "pipeline.h"
#include "graphics_context.h"
#include "hot_reload.h"
#include "../shaders/shared.h"

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

	{ // Initialization pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_init_particles");
		particle_init_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_init_pipeline);
	}

	{ // Simulate pipeline
		ComputePipelineBuilder builder(ctx->device, true);
		builder.set_shader_filepath("gpu_particles.hlsl", "cs_simulate_particles");
		particle_simulate_pipeline = new ComputePipelineAsset(builder);
		AssetCatalog::register_asset(particle_simulate_pipeline);
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

	{ // Particle sbuffer
		size_t particle_buffer_size = particle_capacity * sizeof(GPUParticle);
		BufferDesc desc{};
		desc.size = particle_buffer_size;
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		particle_buffer = ctx->create_buffer(desc);
	}

	{ // Scratch sbuffer
		BufferDesc desc{};
		desc.size = sizeof(uint32_t);
		desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		particles_scratch_buffer = ctx->create_buffer(desc);
	}
}

void GPUParticleSystem::simulate(VkCommandBuffer cmd, float dt)
{
	particles_to_spawn += particle_spawn_rate * dt;

	if (!particles_initialized)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_init_pipeline->pipeline.pipeline);
		DescriptorInfo descriptor_info[] = {
			DescriptorInfo(shader_globals),
			DescriptorInfo(system_globals.buffer),
			DescriptorInfo(particle_buffer.buffer),
			DescriptorInfo(particles_scratch_buffer.buffer),
		};

		vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_init_pipeline->pipeline.descriptor_update_template, 
			particle_init_pipeline->pipeline.layout, 0, descriptor_info);

		vkCmdDispatch(cmd, get_dispatch_size(particle_capacity), 1, 1);

		VkMemoryBarrier memory_barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0, nullptr);

		particles_initialized = true;
	}

	{ // Clear scratch buffer
		vkCmdFillBuffer(cmd, particles_scratch_buffer.buffer, 0, 4, 0);
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

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, particle_simulate_pipeline->pipeline.pipeline);
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(system_globals.buffer),
		DescriptorInfo(particle_buffer.buffer),
		DescriptorInfo(particles_scratch_buffer.buffer),
	};

	vkCmdPushDescriptorSetWithTemplateKHR(cmd, particle_simulate_pipeline->pipeline.descriptor_update_template,
		particle_simulate_pipeline->pipeline.layout, 0, descriptor_info);
	GPUParticlePushConstants pc{};
	pc.delta_time = dt;
	pc.particles_to_spawn = (uint32_t)particles_to_spawn;
	particles_to_spawn -= std::floor(particles_to_spawn);
	vkCmdPushConstants(cmd, particle_simulate_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
	vkCmdDispatch(cmd, get_dispatch_size(particle_capacity), 1, 1);

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

void GPUParticleSystem::render(VkCommandBuffer cmd)
{
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(system_globals.buffer),
		DescriptorInfo(particle_buffer.buffer),
		DescriptorInfo(particles_scratch_buffer.buffer),
	};
	vkCmdPushDescriptorSetWithTemplateKHR(cmd, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);
	vkCmdDraw(cmd, 1, particle_capacity, 0, 0);
}

void GPUParticleSystem::destroy()
{
	render_pipeline->builder.destroy_resources(render_pipeline->pipeline);
	particle_init_pipeline->builder.destroy_resources(particle_init_pipeline->pipeline);
	particle_simulate_pipeline->builder.destroy_resources(particle_simulate_pipeline->pipeline);
	ctx->destroy_buffer(particle_buffer);
	ctx->destroy_buffer(system_globals);
	ctx->destroy_buffer(particles_scratch_buffer);
}
