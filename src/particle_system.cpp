#include "particle_system.h"
#include "graphics_context.h"
#include "pipeline.h"
#include "hot_reload.h"
#include "../shaders/shared.h"

void ParticleSystem::update(float dt)
{
	for (uint32_t i = 0; i < particle_count;)
	{
		Particle& p = particles[i];
		p.position += p.velocity * dt;
		p.lifetime -= dt;
		if (p.lifetime <= 0.0f)
		{
			p = particles[--particle_count];
		}
		else
		{
			++i;
		}
	}

	time_until_spawn -= dt;
	if (time_until_spawn <= 0.0f && particle_count < MAX_PARTICLES)
	{
		Particle& p = particles[particle_count++];
		p.lifetime = 5.0f;
		p.position = position;
		p.velocity = glm::vec3(0.0f, 1.0f, 0.0f);
		p.color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		time_until_spawn += spawn_interval;	
	}
}

void ParticleRenderer::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format)
{
	shader_globals = globals_buffer;

	GraphicsPipelineBuilder builder(ctx->device, true);
	builder
		.set_vertex_shader_filepath("particles.hlsl")
		.set_fragment_shader_filepath("particles.hlsl")
		.set_cull_mode(VK_CULL_MODE_NONE)
		.add_color_attachment(render_target_format)
		.set_depth_format(VK_FORMAT_D32_SFLOAT)
		.set_depth_test(VK_TRUE)
		.set_depth_write(VK_FALSE)
		.set_depth_compare_op(VK_COMPARE_OP_LESS)
		.set_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

	render_pipeline = new GraphicsPipelineAsset(builder);
	AssetCatalog::register_asset(render_pipeline);
}

void ParticleRenderer::shutdown()
{
	render_pipeline->builder.destroy_resources(render_pipeline->pipeline);
}

void ParticleRenderer::render(VkCommandBuffer command_buffer, const ParticleSystem& particle_system)
{
	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals)
	};

	vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);

	for (uint32_t i = 0; i < particle_system.particle_count; ++i)
	{
		const Particle& p = particle_system.particles[i];
		PushCostantsParticles pc{};
		pc.color = p.color;
		pc.position = glm::vec4(p.position, 1.0f);
		vkCmdPushConstants(command_buffer, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
		vkCmdDraw(command_buffer, 1, 1, 0, 0);
	}
}
