#pragma once
#include "defines.h"
#include <vector>
#include <unordered_map>
#include "buffer.h"

#define MAX_PARTICLES 512

struct ParticleSystem;
struct ParticleRenderer;

struct Context;
struct GraphicsPipelineAsset;
struct Texture;
struct TextureCatalog;

struct ParticleRenderSettings;

#define PARTICLE_SYSTEM_DIRECTORY "data/particle_systems"

struct ParticleSystemManager
{
	ParticleSystem* active_system = nullptr;
	std::unordered_map<std::string, ParticleSystem*> catalog;
	const char* directory = nullptr;
	ParticleRenderer* renderer;

	float playback_speed = 1.0f;
	bool paused = false;

	void init(ParticleRenderer* renderer);
	void draw_ui();
	void update(float dt);
	void render(VkCommandBuffer cmd);
};

struct ParticleRenderer
{
	Context* ctx;
	GraphicsPipelineAsset* additive_blend_pipeline;
	GraphicsPipelineAsset* alpha_blend_pipeline;
	VkBuffer shader_globals;
	Buffer renderer_settings;
	// Descriptor pool for particle textures
	VkSampler texture_sampler;

	Texture* white_texture;
	TextureCatalog* texture_catalog;

	void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format);
	void shutdown();
	void render(VkCommandBuffer command_buffer, const ParticleSystem& particle_system);
	void set_render_settings(const ParticleRenderSettings& render_settings);
};

struct Particle
{
	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec3 acceleration;
	glm::vec4 color;
	float lifetime;
	float size;
	float rotation;
	int flipbook_index;
};

#define MAX_NAME_LENGTH 64


enum class EmissionShape
{
	NONE = 0,
	CONE,
	MAX,
};

struct ShapeSettings
{
	EmissionShape shape = EmissionShape::NONE;
	float angle = 0.0f;
	float radius = 0.0f;
	float arc = glm::radians(360.0f);
};

struct ParticleSystem
{
	Particle particles[MAX_PARTICLES];
	uint32_t particle_count = 0;

	char name[64] = { 0 };

	glm::vec3 position = glm::vec3(0.0f);
	
	float duration = 1.0f;
	bool looping = true;
	float lifetime = -1.0f; // Negative = infinite lifetime
	float emission_rate = 10.0f;
	float time = 0.0f;

	ShapeSettings shape_settings;
	glm::vec4 particle_color0 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 particle_color1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
	float initial_speed = 5.0f;
	float gravity_modifier = 0.0f;
	glm::vec2 start_rotation = glm::vec2(0.0f);
	float particle_lifetime = 5.0f;
	glm::vec2 start_size = glm::vec2(0.01f);
	bool random_color = false;


	enum BlendMode
	{
		ADDITIVE = 0,
		ALPHA
	};
	int blend_mode = 0;

	ParticleRenderer* renderer = nullptr;
	const Texture* texture = nullptr;
	bool emission_enabled = false;
	const Texture* emission_map = nullptr;
	glm::vec4 albedo_factor = glm::vec4(1.0, 1.0, 1.0, 1.0);
	glm::vec4 emission_factor = glm::vec4(1.0, 1.0, 1.0, 1.0);
	bool use_flipbook_animation = false;
	bool flipbook_frame_blending = false;
	glm::ivec2 flipbook_size = glm::ivec2(1, 1);
	int flipbook_index = 0;
	float time_until_spawn = 0.0f;

	ParticleSystem(ParticleRenderer* renderer);
	void update(float dt);
	void draw_ui();
	bool save();
	bool load(const char* filepath);
	void reset();
};
