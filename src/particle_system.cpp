#include "particle_system.h"
#include "graphics_context.h"
#include "pipeline.h"
#include "hot_reload.h"
#include "../shaders/shared.h"
#include "random.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_vulkan.h"
#include "texture_catalog.h"
#include <fstream>
#include <sstream>

ParticleSystem::ParticleSystem(ParticleRenderer* renderer)
	: renderer(renderer)
{

}

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
		p.flipbook_index = flipbook_index; //random_int_in_range(0, flipbook_size.x * flipbook_size.y);
		time_until_spawn += 1.0f / emission_rate;	
	}
}

void ParticleSystem::draw_ui()
{
	ImGuiIO& io = ImGui::GetIO();

	static int counter = 0;

	ImGui::Text("Particle system settings");               // Display some text (you can use a format strings too)

	ImGui::InputText("name", name, std::size(name));

	ImGui::DragFloat3("emitter position", glm::value_ptr(position), 0.1f, -1000.0f, 1000.0f);
	ImGui::DragFloat("particle lifetime", &particle_lifetime, 0.1f, 0.0f, 100.0f);
	ImGui::DragFloat("particle size", &particle_size, 0.01f, 0.0f, 100.0f);
	if (ImGui::DragFloat("emission rate", &emission_rate, 0.1f, 0.0f, 1000.0f)) time_until_spawn = 1.0f / emission_rate;
	ImGui::DragFloat("initial speed", &initial_speed, 0.1f, 0.0f, 1000.0f);
	ImGui::DragFloat3("acceleration", glm::value_ptr(acceleration), 0.1f, -100.0f, 100.0f);

	ImGui::SliderAngle("cone angle", &cone_angle, 0.0f, 180.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
	ImGui::ColorEdit4("color 0", (float*)&particle_color0);
	ImGui::ColorEdit4("color 1", (float*)&particle_color1);
	ImGui::Checkbox("randomize color", &random_color);

	if (ImGui::BeginCombo("texture", texture ? texture->name : "NONE"))
	{
		assert(renderer);
		for (const auto& t : renderer->texture_catalog->textures)
		{
			bool is_selected = texture == &t.second;
			if (ImGui::Selectable(t.first.c_str(), is_selected))
				texture = &t.second;
		}

		ImGui::EndCombo();
	}
	if (texture)
	{
		ImGui::Checkbox("use flipbook animation", &use_flipbook_animation);
		ImGui::DragInt2("flipbook size", glm::value_ptr(flipbook_size), 1.0f, 1, 16);
		ImGui::DragInt("flipbook index", &flipbook_index, 1.0f, 0, flipbook_size.x * flipbook_size.y - 1);
	}

	static const char* items[]{ "Additive blend", "Alpha blend"};
	ImGui::Combo("blend mode", &blend_mode, items, IM_ARRAYSIZE(items));


	if (ImGui::Button("Save"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
	{
		save();
	}

	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

}

std::ostream& operator<<(std::ostream& os, const glm::vec4& v)
{
	return os << v.x << " " << v.y << " " << v.z << " " << v.w;
}

std::ostream& operator<<(std::ostream& os, const glm::vec3& v)
{
	return os << v.x << " " << v.y << " " << v.z;
}

std::ostream& operator<<(std::ostream& os, const glm::ivec2& v)
{
	return os << v.x << " " << v.y;
}

std::ostream& operator<<(std::ostream& os, const ParticleSystem& ps)
{
	os << "position: " << ps.position << "\n";
	os << "lifetime: " << ps.lifetime << "\n";
	os << "emission_rate: " << ps.emission_rate << "\n";
	os << "cone_angle: " << ps.cone_angle << "\n";
	os << "particle_color0: " << ps.particle_color0 << "\n";
	os << "particle_color1: " << ps.particle_color1 << "\n";
	os << "initial_speed: " << ps.initial_speed << "\n";
	os << "acceleration: " << ps.acceleration << "\n";
	os << "particle_lifetime: " << ps.particle_lifetime << "\n";
	os << "particle_size: " << ps.particle_size << "\n";
	os << "random_color: " << ps.random_color << "\n";
	os << "use_flipbook_animation: " << ps.use_flipbook_animation << "\n";
	os << "flipbook_size: " << ps.flipbook_size << "\n";
	os << "flipbook_index: " << ps.flipbook_index << "\n";
	os << "name: " << ps.name << "\n";

	if (ps.texture) os << "texture: " << ps.texture->name;

	return os;
}

bool ParticleSystem::save()
{
	if (strlen(name) == 0)
	{
		LOG_ERROR("Failed to save particle system: no name set!");
		return false;
	}

	std::filesystem::path filepath = std::filesystem::path(PARTICLE_SYSTEM_DIRECTORY) / std::filesystem::path(std::string(name) + ".particle_system");
	std::ofstream file(filepath);

	if (!file.is_open()) 
	{
		LOG_ERROR("Failed to write file %s", filepath);
		return false;
	}

	file << *this;
	return true;
}

bool ParticleSystem::load(const char* filepath)
{
	std::fstream file(filepath);
	std::vector<std::string> split;
	int line_count = 0;

	for (std::string line; std::getline(file, line);)
	{
		++line_count;
		split.clear();
		size_t colon = line.find_first_of(':');
		if (colon == std::string::npos)
		{
			LOG_ERROR("Error parsing file %s", filepath);
			return false;
		}

		std::string parameter = line.substr(0, colon);
		std::stringstream ss(line.substr(line.find_first_not_of(' ', colon + 1)));
		for (std::string token; std::getline(ss, token, ' ');)
		{
			split.push_back(token);
		}

		if (parameter == "position")
		{
			if (split.size() != 3)
			{
				LOG_ERROR("Error parsing file %s!", filepath);
				return false;
			}
			for (int i = 0; i < 3; ++i) position[i] = (float)std::atof(split[i].c_str());
		}
		else if (parameter == "lifetime")
		{
			if (split.size() != 1) goto error;
			lifetime = (float)std::atof(split[0].c_str());
		}
		else if (parameter == "emission_rate")
		{
			if (split.size() != 1) goto error;
			emission_rate = (float)std::atof(split[0].c_str());
		}
		else if (parameter == "cone_angle")
		{
			if (split.size() != 1) goto error;
			cone_angle = (float)std::atof(split[0].c_str());
		}
		else if (parameter == "particle_color0")
		{
			if (split.size() != 4) goto error;
			for (int i = 0; i < 4; ++i) particle_color0[i] = (float)std::atof(split[i].c_str());
		}
		else if (parameter == "particle_color1")
		{
			if (split.size() != 4) goto error;
			for (int i = 0; i < 4; ++i) particle_color1[i] = (float)std::atof(split[i].c_str());
		}
		else if (parameter == "initial_speed")
		{
			if (split.size() != 1) goto error;
			initial_speed = (float)std::atof(split[0].c_str());
		}
		else if (parameter == "acceleration")
		{
			if (split.size() != 3) goto error;
			for (int i = 0; i < 3; ++i) acceleration[i] = (float)std::atof(split[i].c_str());
		}
		else if (parameter == "particle_lifetime")
		{
			if (split.size() != 1) goto error;
			particle_lifetime = (float)std::atof(split[0].c_str());
		}
		else if (parameter == "particle_size")
		{
			if (split.size() != 1) goto error;
			particle_size = (float)std::atof(split[0].c_str());
		}
		else if (parameter == "random_color")
		{
			if (split.size() != 1) goto error;
			random_color = (bool)std::atoi(split[0].c_str());
		}
		else if (parameter == "texture")
		{
			texture = renderer->texture_catalog->get_texture(split[0].c_str());
			if (!texture) LOG_ERROR("Failed to find texture %s!", split[0].c_str());
		}
		else if (parameter == "use_flipbook_animation")
		{
			if (split.size() != 1) goto error;
			use_flipbook_animation = (bool)std::atoi(split[0].c_str());
		}
		else if (parameter == "flipbook_size")
		{
			if (split.size() != 2) goto error;
			for (int i = 0; i < 2; ++i) flipbook_size[i] = std::atoi(split[i].c_str());
		}
		else if (parameter == "flipbook_index")
		{
			if (split.size() != 1) goto error;
			flipbook_index = std::atoi(split[0].c_str());
		}
		else if (parameter == "name")
		{
			if (split.size() != 1) goto error;
			strncpy(name, split[0].c_str(), std::size(name) - 1);
		}
	}

	return true;

error:
	LOG_ERROR("Error parsing file %s on line %d!", filepath, line_count);
	return false;
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
	const Texture* tex = particle_system.texture ? particle_system.texture : white_texture;
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
		pc.flipbook_index = p.flipbook_index;
		if (particle_system.use_flipbook_animation)
		{
			int flipbook_offset = std::min((int)((1.0f - pc.normalized_lifetime) * flipbook_range), flipbook_range - 1);
			pc.flipbook_index = (pc.flipbook_index + flipbook_offset) % flipbook_range;
		}

		vkCmdPushConstants(command_buffer, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
		vkCmdDraw(command_buffer, 6, 1, 0, 0);
	}
}

static void reload(ParticleSystemManager& manager)
{
	std::filesystem::path path = manager.directory;
	assert(std::filesystem::exists(path));

	char active_name[MAX_NAME_LENGTH] = { 0 };
	if (manager.active_system)
	{
		strncpy(active_name, manager.active_system->name, std::size(active_name) - 1);
		active_name[MAX_NAME_LENGTH - 1] = 0;
	}

	for (auto& ps : manager.catalog)
	{
		delete ps.second;
	}

	manager.catalog.clear();

	for (const auto& f : std::filesystem::directory_iterator(path))
	{
		if (f.is_regular_file())
		{
			std::filesystem::path extension = f.path().extension();
			if (extension == ".particle_system")
			{
				printf("%s\n", f.path().string().c_str());
				std::string name = f.path().filename().string();

				ParticleSystem* ps = new ParticleSystem(manager.renderer);
				if (ps->load(f.path().string().c_str()))
				{
					manager.catalog.insert(std::make_pair(name, ps));
				}
				else
				{
					LOG_ERROR("Failed to load particle system from %s", f.path().string().c_str());
					delete ps;
				}
			}

		}
	}

	manager.active_system = nullptr;
	if (strlen(active_name) != 0)
	{
		for (const auto& ps : manager.catalog)
		{
			if (strcmp(ps.second->name, active_name) == 0)
			{
				manager.active_system = ps.second;
			}
		}
	}
}

void ParticleSystemManager::init(ParticleRenderer* renderer)
{
	this->renderer = renderer;
	this->directory = PARTICLE_SYSTEM_DIRECTORY;

	reload(*this);
}

void ParticleSystemManager::draw_ui()
{
	ImGui::Begin("Particle editor");

	if (ImGui::Button("Reload"))
	{
		reload(*this);
	}
	if (ImGui::BeginCombo("Select particle system", active_system ? active_system->name : "NONE"))
	{
		for (const auto& ps : catalog)
		{
			if (ImGui::Selectable(ps.first.c_str(), active_system == ps.second))
			{
				active_system = ps.second;
			}
		}

		ImGui::EndCombo();
	}

	if (active_system)
	{
		active_system->draw_ui();
	}

	ImGui::End();
}

void ParticleSystemManager::update(float dt)
{
	if (active_system)
		active_system->update(dt);
}

void ParticleSystemManager::render(VkCommandBuffer cmd)
{
	if (renderer && active_system)
	{
		renderer->render(cmd, *active_system);
	}
}

