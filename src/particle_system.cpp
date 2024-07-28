#include "particle_system.h"
#include "graphics_context.h"
#include "pipeline.h"
#include "hot_reload.h"
#include "../shaders/shared.h"
#include "random.h"
#include "imgui/imgui.h"

void ParticleSystem::update(float dt)
{
	for (uint32_t i = 0; i < particle_count;)
	{
		Particle& p = particles[i];
		p.velocity += p.acceleration * dt;
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
		p.lifetime = particle_lifetime;
		p.position = position;
		p.velocity = random_vector_in_oriented_cone(cosf(cone_angle), glm::vec3(0.0, 1.0f, 0.0f)) * initial_speed;
		p.color = random_color ? random_vector<glm::vec4>() : particle_color;
		p.acceleration = acceleration;
		p.size = particle_size;
		time_until_spawn += 1.0f / emission_rate;	
	}
}

void ParticleSystem::draw_ui()
{
	ImGuiIO& io = ImGui::GetIO();

	static int counter = 0;

	ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

	ImGui::Text("Particle system settings");               // Display some text (you can use a format strings too)

	ImGui::DragFloat3("emitter position", glm::value_ptr(position), 0.1f, -1000.0f, 1000.0f);
	ImGui::DragFloat("particle lifetime", &particle_lifetime, 0.1f, 0.0f, 100.0f);
	ImGui::DragFloat("emission rate", &emission_rate, 0.1f, 0.0f, 1000.0f);
	ImGui::DragFloat("initial speed", &initial_speed, 0.1f, 0.0f, 1000.0f);
	ImGui::DragFloat3("acceleration", glm::value_ptr(acceleration), 0.1f, -100.0f, 100.0f);

	ImGui::SliderAngle("cone angle", &cone_angle, 0.0f, 180.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
	ImGui::ColorEdit4("particle color", (float*)&particle_color); // Edit 3 floats representing a color
	ImGui::Checkbox("randomize color", &random_color);

	if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
		counter++;
	ImGui::SameLine();
	ImGui::Text("counter = %d", counter);

	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

	ImGui::End();
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
		.set_blend_preset(BlendPreset::ADDITIVE)
		.set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

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
		pc.size = p.size;
		pc.normalized_lifetime = p.lifetime;
		vkCmdPushConstants(command_buffer, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
		vkCmdDraw(command_buffer, 6, 1, 0, 0);
	}
}
