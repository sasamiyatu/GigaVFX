#pragma once
#include "defines.h"
#include "buffer.h"
#include "radix_sort.h"

struct GPUParticleSystem
{
    void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity);
    void simulate(VkCommandBuffer cmd, float dt, struct CameraState& camera_state);
    void render(VkCommandBuffer cmd);
    void destroy();
    void draw_stats_overlay();
    void draw_ui(); // Draws into the currently active imgui window

    struct Context* ctx = nullptr;
    VkBuffer shader_globals = VK_NULL_HANDLE;
    Buffer system_globals = {};

    // Double buffered
    Buffer particle_buffer[2] = {};
    Buffer particle_system_state[2] = {};

    Buffer indirect_dispatch_buffer = {};

    Buffer sort_keyval_buffer[2] = {}; // Radix sort uses two buffers internally
    Buffer sort_internal_buffer = {};
    Buffer sort_indirect_buffer = {};

    VkQueryPool query_pool = VK_NULL_HANDLE;

    struct GraphicsPipelineAsset* render_pipeline = nullptr;
    struct ComputePipelineAsset* particle_emit_pipeline = nullptr;
    struct ComputePipelineAsset* particle_dispatch_size_pipeline = nullptr;
    struct ComputePipelineAsset* particle_simulate_pipeline = nullptr;
    struct ComputePipelineAsset* particle_compact_pipeline = nullptr;
    struct ComputePipelineAsset* particle_debug_sort_pipeline = nullptr;
    uint32_t particle_capacity = 0;
    float particle_spawn_rate = 1.0f;
    float particles_to_spawn = 0.0f;
    bool particles_initialized = false;
    float particle_size = 0.1f; // World space
    glm::vec4 particle_color = glm::vec4(1.0f);
    glm::vec3 particle_sort_axis = glm::vec3(1.0f, 0.0f, 0.0f);

    RadixSortContext* sort_context = nullptr;

    struct
    {
        double simulate_total = 0.0;
        double render_total = 0.0;
    } performance_timings;
};