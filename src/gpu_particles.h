#pragma once
#include "defines.h"
#include "buffer.h"

struct GPUParticleSystem
{
    void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity);
    void simulate(VkCommandBuffer cmd, float dt);
    void render(VkCommandBuffer cmd);
    void destroy();
    void draw_stats_overlay();

    struct Context* ctx = nullptr;
    VkBuffer shader_globals = VK_NULL_HANDLE;
    Buffer system_globals = {};

    // Double buffered
    Buffer particle_buffer[2] = {};
    Buffer particle_system_state[2] = {};
    Buffer indirect_dispatch_buffer = {};

    VkQueryPool query_pool = VK_NULL_HANDLE;

    struct GraphicsPipelineAsset* render_pipeline = nullptr;
    struct ComputePipelineAsset* particle_emit_pipeline = nullptr;
    struct ComputePipelineAsset* particle_dispatch_size_pipeline = nullptr;
    struct ComputePipelineAsset* particle_simulate_pipeline = nullptr;
    struct ComputePipelineAsset* particle_compact_pipeline = nullptr;
    uint32_t particle_capacity = 0;
    float particle_spawn_rate = 10000.0f;
    float particles_to_spawn = 0.0f;
    bool particles_initialized = false;

    struct
    {
        double simulate_total = 0.0;
        double render_total = 0.0;
    } performance_timings;
};