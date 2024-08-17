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

constexpr glm::vec3 GRAVITY = glm::vec3(0.0f, -9.81f, 0.0f);

ParticleSystem::ParticleSystem(ParticleRenderer* renderer)
	: renderer(renderer)
{

}

void ParticleSystem::update(float dt)
{
	if (lifetime <= 0.0f) return;
	
	lifetime -= dt;
	while (lifetime <= 0.0f && looping) lifetime += duration;

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
	if (time_until_spawn < 0.0f && particle_count < MAX_PARTICLES)
	{
		Particle& p = particles[particle_count++];
		p.lifetime = particle_lifetime;
		switch (shape_settings.shape)
		{
		case EmissionShape::NONE:
			p.velocity = glm::vec3(0.0f, 1.0f, 0.0f) * initial_speed;
			p.position = position;
			break;
		case EmissionShape::CONE:
		{
			p.velocity = random_vector_in_oriented_cone(cosf(shape_settings.angle), glm::vec3(0.0, 1.0f, 0.0f)) * initial_speed;
			float arc = random_in_range(0.0f, shape_settings.arc);
			float r = random_in_range(0.0f, shape_settings.radius);
			p.position = position + glm::vec3(cosf(arc), 0.0f, sinf(arc)) * r;
		} break;
		default:
			assert(false);
			break;
		}
		p.color = random_color ? random_vector<glm::vec4>() : glm::lerp(particle_color0, particle_color1, uniform_random());
		p.acceleration = GRAVITY * gravity_modifier;
		p.size = random_in_range(start_size.x, start_size.y);
		p.flipbook_index = flipbook_index; //random_int_in_range(0, flipbook_size.x * flipbook_size.y);
		p.rotation = glm::radians(random_in_range(start_rotation.x, start_rotation.y));
		time_until_spawn += 1.0f / emission_rate;	
	}
}

static void set_renderer_settings(ParticleSystem& ps)
{
	ParticleRenderSettings settings{};
	settings.albedo_multiplier = ps.albedo_factor;
	settings.emission_multiplier = ps.emission_enabled ? ps.emission_factor : glm::vec4(0.0f);
	ps.renderer->set_render_settings(settings);
}

void ParticleSystem::draw_ui()
{
	ImGuiIO& io = ImGui::GetIO();

	static int counter = 0;

	ImGui::Text("Particle system settings");               // Display some text (you can use a format strings too)

	ImGui::InputText("name", name, std::size(name));
	
	ImGui::DragFloat("duration", &duration);
	ImGui::Checkbox("looping", &looping);
	ImGui::DragFloat3("emitter position", glm::value_ptr(position), 0.1f, -1000.0f, 1000.0f);
	ImGui::DragFloat("particle lifetime", &particle_lifetime, 0.1f, 0.0f, 100.0f);
	ImGui::DragFloat2("start size", glm::value_ptr(start_size), 0.01f, 0.0f, 100.0f);
	if (ImGui::DragFloat("emission rate", &emission_rate, 0.1f, 0.0f, 1000.0f)) time_until_spawn = 1.0f / emission_rate;
	ImGui::DragFloat("initial speed", &initial_speed, 0.1f, 0.0f, 1000.0f);
	ImGui::DragFloat("gravity_modifier", &gravity_modifier, 0.1f, 0.0f, 100.0f);
	ImGui::DragFloat2("start rotation", glm::value_ptr(start_rotation), 0.0f, 360.0f);

	if (ImGui::CollapsingHeader("Shape"))
	{
		const char* names[(int)EmissionShape::MAX] = { "None", "Cone" };
		if (ImGui::BeginCombo("Select shape", names[(int)shape_settings.shape]))
		{
			for (int i = 0; i < std::size(names); ++i)
			{
				if (ImGui::Selectable(names[i], (int)shape_settings.shape == i))
				{
					shape_settings.shape = (EmissionShape)i;
				}
			}

			ImGui::EndCombo();
		}

		switch (shape_settings.shape)
		{
		case EmissionShape::CONE:
			ImGui::SliderAngle("Angle", &shape_settings.angle, 0.0f, 90.0f);
			ImGui::DragFloat("Radius", &shape_settings.radius, 0.1f, 0.0f, 3000.0f);
			ImGui::SliderAngle("Arc", &shape_settings.arc, 0.0f, 360.0f);
			break;
		default:
			break;
		}
	}
	ImGui::ColorEdit4("color 0", (float*)&particle_color0);
	ImGui::ColorEdit4("color 1", (float*)&particle_color1);
	ImGui::Checkbox("randomize color", &random_color);

	if (ImGui::CollapsingHeader("Rendering"))
	{
		if (ImGui::BeginCombo("Albedo", texture ? texture->name : "NONE"))
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
		bool color_changed = false;
		color_changed |= ImGui::ColorEdit4("Albedo factor", glm::value_ptr(albedo_factor));
		color_changed |= ImGui::Checkbox("Emission", &emission_enabled);
		if (ImGui::BeginCombo("Emission map", emission_map ? emission_map->name : "NONE"))
		{
			assert(renderer);
			for (const auto& t : renderer->texture_catalog->textures)
			{
				bool is_selected = emission_map == &t.second;
				if (ImGui::Selectable(t.first.c_str(), is_selected))
					emission_map = &t.second;
			}

			ImGui::EndCombo();
		}
		color_changed |= ImGui::ColorEdit4("Emission factor", glm::value_ptr(emission_factor));
		if (color_changed) set_renderer_settings(*this);
		if (texture)
		{
			ImGui::Checkbox("use flipbook animation", &use_flipbook_animation);
			ImGui::DragInt2("flipbook size", glm::value_ptr(flipbook_size), 1.0f, 1, 16);
			ImGui::DragInt("flipbook index", &flipbook_index, 1.0f, 0, flipbook_size.x * flipbook_size.y - 1);
			ImGui::Checkbox("flipbook frame blending", &flipbook_frame_blending);
		}

		static const char* items[]{ "Additive blend", "Alpha blend" };
		ImGui::Combo("blend mode", &blend_mode, items, IM_ARRAYSIZE(items));
	}

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

std::ostream& operator<<(std::ostream& os, const glm::vec2& v)
{
	return os << v.x << " " << v.y;
}

std::ostream& operator<<(std::ostream& os, const glm::ivec2& v)
{
	return os << v.x << " " << v.y;
}

std::ostream& operator<<(std::ostream& os, const ParticleSystem& ps)
{
	os << "position: " << ps.position << "\n";
	os << "emission_rate: " << ps.emission_rate << "\n";
	os << "cone_angle: " << ps.shape_settings.angle << "\n";
	os << "particle_color0: " << ps.particle_color0 << "\n";
	os << "particle_color1: " << ps.particle_color1 << "\n";
	os << "initial_speed: " << ps.initial_speed << "\n";
	os << "gravity_modifier: " << ps.gravity_modifier << "\n";
	os << "particle_lifetime: " << ps.particle_lifetime << "\n";
	os << "start_size: " << ps.start_size << "\n";
	os << "random_color: " << ps.random_color << "\n";
	os << "use_flipbook_animation: " << ps.use_flipbook_animation << "\n";
	os << "flipbook_size: " << ps.flipbook_size << "\n";
	os << "flipbook_index: " << ps.flipbook_index << "\n";
	os << "name: " << ps.name << "\n";
	os << "emission: " << ps.emission_enabled << "\n";
	os << "albedo_factor: " << ps.albedo_factor << "\n";
	os << "emission_factor: " << ps.emission_factor << "\n";
	os << "blend_mode: " << ps.blend_mode << "\n";
	os << "flipbook_frame_blending: " << ps.flipbook_frame_blending << "\n";
	os << "duration: " << ps.duration << "\n";
	os << "looping: " << ps.looping << "\n";
	os << "start_rotation: " << ps.start_rotation << "\n";
	os << "shape: " << (int)ps.shape_settings.shape << "\n";
	os << "arc: " << ps.shape_settings.arc << "\n";
 
	if (ps.texture) os << "texture: " << ps.texture->name << "\n";
	if (ps.emission_map) os << "emission_map: " << ps.emission_map->name << "\n";

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

static void read_floats(const std::vector<std::string>& tokens, const char* parameter_name, float* out_floats, size_t count)
{
	if (tokens.size() != count)
	{
		LOG_ERROR("Error parsing parameter %s", parameter_name);
	}

	for (size_t i = 0; i < count; ++i) out_floats[i] = (float)std::atof(tokens[i].c_str());
}

#define READ_FLOATS(var, name, count)                           \
if (parameter == name)                                          \
{																\
	float* addr = (float*)&var;                                 \
	for (size_t i = 0; i < count; ++i)                          \
	{                                                           \
		addr[i] = std::atof(split[i].c_str());                  \
	}                                                           \
	continue;													\
}

#define READ_INTS(var, name, count)                             \
if (parameter == name)                                          \
{																\
	int* addr = (int*)&var;										\
	for (size_t i = 0; i < count; ++i)                          \
	{                                                           \
		addr[i] = std::atoi(split[i].c_str());                  \
	}                                                           \
	continue;													\
}

#define READ_BOOL(var, name)									\
if (parameter == name)                                          \
{																\
	var = (bool)std::atoi(split[0].c_str());                    \
	continue;													\
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

		READ_FLOATS(position, "position", 3);
		READ_FLOATS(emission_rate, "emission_rate", 1);
		READ_FLOATS(shape_settings.angle, "cone_angle", 1);
		READ_FLOATS(particle_color0, "particle_color0", 4);
		READ_FLOATS(particle_color1, "particle_color1", 4);
		READ_FLOATS(initial_speed, "initial_speed", 1);
		READ_FLOATS(gravity_modifier, "gravity_modifier", 1);
		READ_FLOATS(particle_lifetime, "particle_lifetime", 1);
		READ_FLOATS(start_size, "start_size", 2);
		READ_FLOATS(albedo_factor, "albedo_factor", 4);
		READ_FLOATS(emission_factor, "emission_factor", 4);
		READ_FLOATS(duration, "duration", 1);
		READ_FLOATS(start_rotation, "start_rotation", 2);
		READ_FLOATS(shape_settings.arc, "arc", 1);
		READ_INTS(flipbook_size, "flipbook_size", 2);
		READ_INTS(flipbook_index, "flipbook_index", 1);
		READ_INTS(blend_mode, "blend_mode", 1);
		READ_INTS(shape_settings.shape, "shape", 1);
		READ_BOOL(random_color, "random_color");
		READ_BOOL(emission_enabled, "emission");
		READ_BOOL(flipbook_frame_blending, "flipbook_frame_blending");
		READ_BOOL(use_flipbook_animation, "use_flipbook_animation");
		READ_BOOL(looping, "looping");
		if (parameter == "texture")
		{
			texture = renderer->texture_catalog->get_texture(split[0].c_str());
			if (!texture) LOG_ERROR("Failed to find texture %s!", split[0].c_str());
		}
		else if (parameter == "name")
		{
			if (split.size() != 1) goto error;
			strncpy(name, split[0].c_str(), std::size(name) - 1);
		}
		else if (parameter == "emission_map")
		{
			if (split.size() != 1) goto error;
			emission_map = renderer->texture_catalog->get_texture(split[0].c_str());
			if (!emission_map) LOG_ERROR("Failed to find texture %s!", split[0].c_str());
		}
	}

	reset();

	return true;

error:
	LOG_ERROR("Error parsing file %s on line %d!", filepath, line_count);
	return false;
}

void ParticleSystem::reset()
{
	particle_count = 0;
	time_until_spawn = 0.0f;
	lifetime = duration;
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

	BufferDesc desc{};
	ParticleRenderSettings default_settings{};
	default_settings.albedo_multiplier = glm::vec4(1.0f);
	default_settings.emission_multiplier = glm::vec4(0.0f);
	desc.size = sizeof(ParticleRenderSettings);
	desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	desc.usage_flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	desc.data = &default_settings;
	renderer_settings = ctx->create_buffer(desc);
}

void ParticleRenderer::shutdown()
{
	additive_blend_pipeline->builder.destroy_resources(additive_blend_pipeline->pipeline);
	alpha_blend_pipeline->builder.destroy_resources(alpha_blend_pipeline->pipeline);
	vkDestroySampler(ctx->device, texture_sampler, nullptr);
	white_texture->destroy(ctx->device, ctx->allocator);
	vmaDestroyBuffer(ctx->allocator, renderer_settings.buffer, renderer_settings.allocation);
}

void ParticleRenderer::render(VkCommandBuffer command_buffer, const ParticleSystem& particle_system)
{
	GraphicsPipelineAsset* render_pipeline = particle_system.blend_mode == ParticleSystem::ADDITIVE ? additive_blend_pipeline : alpha_blend_pipeline;
	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline->pipeline.pipeline);
	const Texture* tex = particle_system.texture ? particle_system.texture : white_texture;
	const Texture* emissive = particle_system.emission_map ? particle_system.emission_map : white_texture;
	DescriptorInfo descriptor_info[] = {
		DescriptorInfo(shader_globals),
		DescriptorInfo(renderer_settings.buffer),
		DescriptorInfo(texture_sampler),
		DescriptorInfo(tex->view, tex->layout),
		DescriptorInfo(emissive->view, emissive->layout),
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
		float age = 1.0f - pc.normalized_lifetime;
		pc.flipbook_index0 = pc.flipbook_index1 = p.flipbook_index;
		pc.rotation = p.rotation;
		if (particle_system.use_flipbook_animation)
		{
			float lerp = glm::fract(age * (float)flipbook_range);
			int flipbook_offset = std::min((int)(age * flipbook_range), flipbook_range - 1);
			pc.flipbook_index0 = (p.flipbook_index + flipbook_offset) % flipbook_range;
			if (particle_system.flipbook_frame_blending)
			{
				pc.flipbook_index1 = std::min((int)pc.flipbook_index0 + 1, flipbook_range - 1);
				pc.flipbook_blend = lerp;
			}
			else
			{
				pc.flipbook_index1 = pc.flipbook_index0;
				pc.flipbook_blend = 0.0f;
			}
		}

		vkCmdPushConstants(command_buffer, render_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
		vkCmdDraw(command_buffer, 6, 1, 0, 0);
	}
}

void ParticleRenderer::set_render_settings(const ParticleRenderSettings& settings)
{
	void* mapped = nullptr;
	vmaMapMemory(ctx->allocator, renderer_settings.allocation, &mapped);
	memcpy(mapped, &settings, sizeof(settings));
	vmaUnmapMemory(ctx->allocator, renderer_settings.allocation);
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
				set_renderer_settings(*active_system);
			}
		}

		ImGui::EndCombo();
	}

	if (active_system)
	{
		active_system->draw_ui();
	}

	ImGui::End();

	ImGui::Begin("Particle simulation");

	if (paused)
	{
		if (ImGui::Button("Play"))
		{
			paused = false;
		}
	}
	else
	{
		if (ImGui::Button("Pause"))
		{
			paused = true;
		}
	}

	ImGui::SameLine();

	if (ImGui::Button("Restart"))
	{
		if (active_system)
			active_system->reset();
	}

	ImGui::SameLine();

	if (ImGui::Button("Stop"))
	{
		if (active_system)
			active_system->reset();
		paused = true;
	}

	ImGui::DragFloat("Playback Speed", &playback_speed, 0.1f, 0.0f, 10.0f);
	if (active_system)
	{
		ImGui::Text("Playback time: %f", active_system->duration - active_system->lifetime);
	}

	ImGui::End();
}

void ParticleSystemManager::update(float dt)
{
	float t = !paused ? dt * playback_speed : 0.0f;
	if (active_system)
		active_system->update(t);
}

void ParticleSystemManager::render(VkCommandBuffer cmd)
{
	if (renderer && active_system)
	{
		renderer->render(cmd, *active_system);
	}
}

