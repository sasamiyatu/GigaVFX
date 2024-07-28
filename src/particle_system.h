#pragma once
#include "defines.h"
#include <vector>

#define MAX_PARTICLES 512

struct Particle
{
	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec4 color;
	float lifetime;
};

struct ParticleSystem
{
	Particle particles[MAX_PARTICLES];
	uint32_t particle_count = 0;

	glm::vec3 position = glm::vec3(0.0f);

	float lifetime = -1.0f; // Negative = infinite lifetime
	float spawn_interval = 0.1f;
	float time = 0.0f;
	float time_until_spawn = 1.0f;

	void update(float dt);
};

struct Context;
struct GraphicsPipelineAsset;

struct ParticleRenderer
{
	Context* ctx;
	GraphicsPipelineAsset* render_pipeline;
	VkBuffer shader_globals;

	void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format);
	void shutdown();
	void render(VkCommandBuffer command_buffer, const ParticleSystem& particle_system);
};