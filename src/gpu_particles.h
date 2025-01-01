#pragma once
#include "defines.h"
#include "buffer.h"
#include "radix_sort.h"

struct AccelerationStructure
{
    VkAccelerationStructureKHR acceleration_structure;
    Buffer acceleration_structure_buffer;
    Buffer scratch_buffer;
};

struct GPUParticleSystem
{
    void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity, const Texture& shadowmap_texture);
    void simulate(VkCommandBuffer cmd, float dt, struct CameraState& camera_state, glm::vec3 light_dir);
    void render(VkCommandBuffer cmd, const Texture& render_target, const Texture& depth_target);
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
    Buffer indirect_draw_buffer = {};

    Buffer sort_keyval_buffer[2] = {}; // Radix sort uses two buffers internally
    Buffer sort_internal_buffer = {};
    Buffer sort_indirect_buffer = {};

    VkQueryPool query_pool = VK_NULL_HANDLE;

    struct GraphicsPipelineAsset* render_pipeline_back_to_front = nullptr;
    struct GraphicsPipelineAsset* render_pipeline_front_to_back = nullptr;
    struct GraphicsPipelineAsset* render_pipeline_light = nullptr;
    struct ComputePipelineAsset* particle_emit_pipeline = nullptr;
    struct ComputePipelineAsset* particle_dispatch_size_pipeline = nullptr;
    struct ComputePipelineAsset* particle_draw_count_pipeline = nullptr;
    struct ComputePipelineAsset* particle_simulate_pipeline = nullptr;
    struct ComputePipelineAsset* particle_compact_pipeline = nullptr;
    struct ComputePipelineAsset* particle_debug_sort_pipeline = nullptr;
    struct ComputePipelineAsset* particle_composite_pipeline = nullptr;
    uint32_t particle_capacity = 0;
    float particle_spawn_rate = 10000.0f;
    float particles_to_spawn = 0.0f;
    bool particles_initialized = false;
    float particle_size = 0.05f; // World space
    glm::vec4 particle_color = glm::vec4(glm::vec3(1.0f), 0.2f);
    glm::vec3 particle_sort_axis = glm::vec3(1.0f, 0.0f, 0.0f);
    float particle_lifetime = 3.0f;
    float noise_scale = 1.0f;
    float noise_time_scale = 1.0f;
    bool sort_particles = true;
    uint32_t num_slices = 64;
    int slices_to_display = 64;
    bool display_single_slice = false;
    float shadow_alpha = 0.2f;
    bool draw_order_flipped = false;
    float emitter_radius = 0.1f;
    float particle_speed = 0.5f;
    float time = 0.0f;
    glm::vec3 color_attenuation = glm::vec3(1.0f);
    uint32_t light_buffer_size;

    Texture particle_render_target;
    Texture light_render_target;
    VkSampler light_sampler;

    RadixSortContext* sort_context = nullptr;

    AccelerationStructure blas = {};
    Buffer particle_aabbs = {}; // Acceleration structure input
    AccelerationStructure tlas = {};
    Buffer instances_buffer = {};

    VkImageView light_depth_view = VK_NULL_HANDLE;

    struct
    {
        double simulate_total = 0.0;
        double render_total = 0.0;
    } performance_timings;
};