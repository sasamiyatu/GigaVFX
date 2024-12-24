#pragma once
#include "defines.h"
#include "buffer.h"

struct GPUParticleSystem
{
    void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity);
    void simulate(VkCommandBuffer cmd, float dt);
    void render(VkCommandBuffer cmd);
    void destroy();

    struct Context* ctx = nullptr;
    VkBuffer shader_globals = VK_NULL_HANDLE;
    Buffer system_globals = {};
    Buffer particle_buffer = {};
    Buffer particles_scratch_buffer = {};
    struct GraphicsPipelineAsset* render_pipeline = nullptr;
    struct ComputePipelineAsset* particle_init_pipeline = nullptr;
    struct ComputePipelineAsset* particle_simulate_pipeline = nullptr;
    uint32_t particle_capacity = 0;
    float particle_spawn_rate = 1.0f;
    float particles_to_spawn = 0.0f;
    bool particles_initialized = false;
};