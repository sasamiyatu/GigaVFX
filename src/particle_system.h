#pragma once
#include "defines.h"
#include <vector>

#define MAX_PARTICLES 512

struct ParticleSystem;

struct Context;
struct GraphicsPipelineAsset;
struct Texture;
struct TextureCatalog;

struct ParticleRenderer
{
	Context* ctx;
	GraphicsPipelineAsset* additive_blend_pipeline;
	GraphicsPipelineAsset* alpha_blend_pipeline;
	VkBuffer shader_globals;
	// Descriptor pool for particle textures
	VkSampler texture_sampler;

	Texture* white_texture;
	TextureCatalog* texture_catalog;

	void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format);
	void shutdown();
	void render(VkCommandBuffer command_buffer, const ParticleSystem& particle_system);
};

struct Particle
{
	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec3 acceleration;
	glm::vec4 color;
	float lifetime;
	float size;
	int flipbook_index;
};

struct ParticleSystem
{
	Particle particles[MAX_PARTICLES];
	uint32_t particle_count = 0;

	glm::vec3 position = glm::vec3(0.0f);

	float lifetime = -1.0f; // Negative = infinite lifetime
	float emission_rate = 10.0f;
	float time = 0.0f;

	float cone_angle = 0.0f;
	glm::vec4 particle_color0 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 particle_color1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
	float initial_speed = 5.0f;
	glm::vec3 acceleration = glm::vec3(0.0f, -9.81f, 0.0f);
	float particle_lifetime = 5.0f;
	float particle_size = 0.01f;
	bool random_color = false;

	enum BlendMode
	{
		ADDITIVE = 0,
		ALPHA
	};
	int blend_mode = 0;

	ParticleRenderer* renderer = nullptr;
	const Texture* texture = nullptr;
	bool use_flipbook_animation = false;
	glm::ivec2 flipbook_size = glm::ivec2(1, 1);
	int flipbook_index = 0;

	float time_until_spawn = 0.0f;

	void update(float dt);

	void draw_ui();
};
