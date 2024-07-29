#include "particle_system.h"
#include "graphics_context.h"
#include "pipeline.h"
#include "hot_reload.h"
#include "../shaders/shared.h"
#include "random.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_vulkan.h"

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
		p.color = random_color ? random_vector<glm::vec4>() : glm::lerp(particle_color0, particle_color1, uniform_random());
		p.acceleration = acceleration;
		p.size = particle_size;
		p.flipbook_index = random_int_in_range(0, flipbook_size.x * flipbook_size.y);
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
	ImGui::DragFloat("particle size", &particle_size, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("emission rate", &emission_rate, 0.1f, 0.0f, 1000.0f);
	ImGui::DragFloat("initial speed", &initial_speed, 0.1f, 0.0f, 1000.0f);
	ImGui::DragFloat3("acceleration", glm::value_ptr(acceleration), 0.1f, -100.0f, 100.0f);

	ImGui::SliderAngle("cone angle", &cone_angle, 0.0f, 180.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
	ImGui::ColorEdit4("color 0", (float*)&particle_color0);
	ImGui::ColorEdit4("color 1", (float*)&particle_color1);
	ImGui::Checkbox("randomize color", &random_color);

	if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
		counter++;
	ImGui::SameLine();
	ImGui::Text("counter = %d", counter);

	if (texture)
	{
		static bool show_texture_preview = false;
		if (ImGui::Button("Texture preview"))
			show_texture_preview = true;

		if (show_texture_preview)
		{
			ImGui::Begin("Texture preview", &show_texture_preview);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
			glm::vec2 uv_scale = glm::vec2(1.0f / glm::vec2(flipbook_size));
			glm::vec2 uv_offset = glm::vec2(uv_scale.x * (flipbook_index % flipbook_size.x), uv_scale.y * (flipbook_index / flipbook_size.x));
			glm::vec2 uv0 = uv_offset;
			glm::vec2 uv1 = uv_offset + uv_scale;
			ImGui::Image(texture->descriptor_set, ImVec2(texture->texture->width, texture->texture->height), ImVec2(uv0.x, uv0.y), ImVec2(uv1.x, uv1.y));
			if (ImGui::Button("Close"))
				show_texture_preview = false;
			ImGui::End();
		}

		ImGui::DragInt2("flipbook size", glm::value_ptr(flipbook_size), 1.0f, 1, 16);
		ImGui::DragInt("flipbook index", &flipbook_index, 1.0f, 0, flipbook_size.x * flipbook_size.y - 1);
	}

	static const char* items[]{ "Additive blend", "Alpha blend"};
	ImGui::Combo("blend mode", &blend_mode, items, IM_ARRAYSIZE(items));

	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

	ImGui::End();
}

void ParticleRenderer::init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format)
{
	shader_globals = globals_buffer;
	this->ctx = ctx;

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

	{
		{
			VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
			info.magFilter = VK_FILTER_LINEAR;
			info.minFilter = VK_FILTER_LINEAR;
			info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			info.maxLod = VK_LOD_CLAMP_NONE;
			info.anisotropyEnable = VK_TRUE;
			info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			info.maxAnisotropy = ctx->physical_device.properties.limits.maxSamplerAnisotropy;
			VK_CHECK(vkCreateSampler(ctx->device, &info, nullptr, &texture_sampler));
		}
	}

	{
		white_texture = new Texture();
		memset(white_texture, 0, sizeof(Texture));

		white_texture->source = (uint8_t*)malloc(4);
		assert(white_texture->source);
		memset(white_texture->source, 0xFF, 4);
		white_texture->width = 1;
		white_texture->height = 1;

		ctx->create_textures(white_texture, 1);
	}

	additive_blend_pipeline = new GraphicsPipelineAsset(builder);
	builder.set_blend_preset(BlendPreset::ALPHA);
	alpha_blend_pipeline = new GraphicsPipelineAsset(builder);
	AssetCatalog::register_asset(additive_blend_pipeline);
	AssetCatalog::register_asset(alpha_blend_pipeline);
}

void ParticleRenderer::shutdown()
{
	additive_blend_pipeline->builder.destroy_resources(additive_blend_pipeline->pipeline);
	alpha_blend_pipeline->builder.destroy_resources(alpha_blend_pipeline->pipeline);
	vkDestroySampler(ctx->device, texture_sampler, nullptr);
	white_texture->destroy(ctx->device, ctx->allocator);
}

void ParticleRenderer::render(VkCommandBuffer command_buffer, const ParticleSystem& particle_system)
{
	GraphicsPipelineAsset* render_pipeline = particle_system.blend_mode == ParticleSystem::ADDITIVE ? additive_blend_pipeline : alpha_blend_pipeline;
	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);
	const Texture* tex = particle_system.texture ? particle_system.texture->texture : white_texture;
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(texture_sampler),
		DescriptorInfo(tex->view, tex->layout),
	};

	vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, render_pipeline->pipeline.descriptor_update_template, render_pipeline->pipeline.layout, 0, descriptor_info);

	int flipbook_range = particle_system.flipbook_size.x * particle_system.flipbook_size.y;
	const float inv_particle_lifetime = 1.0f / particle_system.particle_lifetime;
	for (uint32_t i = 0; i < particle_system.particle_count; ++i)
	{
		const Particle& p = particle_system.particles[i];

		PushCostantsParticles pc{};
		pc.color = p.color;
		pc.position = glm::vec4(p.position, 1.0f);
		pc.flipbook_size = particle_system.flipbook_size;
		pc.size = p.size;
		pc.normalized_lifetime = glm::clamp(p.lifetime * inv_particle_lifetime, 0.0f, 1.0f);
		int flipbook_offset = std::min((int)(pc.normalized_lifetime * flipbook_range), flipbook_range - 1);
		pc.flipbook_index = (p.flipbook_index + flipbook_offset) % flipbook_range;
		vkCmdPushConstants(command_buffer, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
		vkCmdDraw(command_buffer, 6, 1, 0, 0);
	}
}

void ParticleRenderer::add_texture(const Texture* tex)
{
	VkDescriptorSet descriptor_set = ImGui_ImplVulkan_AddTexture(texture_sampler, tex->view, tex->layout);
	ParticleTexture pt{};
	pt.texture = tex;
	pt.descriptor_set = descriptor_set;
	textures.push_back(pt);
}
