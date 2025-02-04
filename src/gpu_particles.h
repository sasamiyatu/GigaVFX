#pragma once
#include "defines.h"
#include "buffer.h"
#include "radix_sort.h"
#include "sdf.h"
#include "pipeline.h"
#include "../shaders/shared.h"

struct AccelerationStructure
{
    VkAccelerationStructureKHR acceleration_structure;
    Buffer acceleration_structure_buffer;
    Buffer scratch_buffer;
};

struct ShaderInfo
{
    std::string shader_source_file;
    std::string entry_point;
};

struct SDF;

struct IConfigUI
{
    virtual void draw_config_ui() {}
    virtual const char* get_display_name() { return "NONAME"; }
};

struct GPUParticleSystem : IConfigUI
{
    void init(struct Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, uint32_t particle_capacity,
        const Texture& shadowmap_texture, uint32_t cascade_index, const ShaderInfo& emit_shader, const ShaderInfo& update_shader,
        bool emit_once = false);
    void simulate(VkCommandBuffer cmd, float dt, struct CameraState& camera_state, glm::mat4 shadow_view, glm::mat4 shadow_projection);
    void render(VkCommandBuffer cmd, const Texture& depth_target);
    void composite(VkCommandBuffer cmd, const Texture& render_target);
    void destroy();
    void draw_stats_overlay();
    void set_position(glm::vec3 pos) { position = pos; }

    // IConfigUI
    virtual void draw_config_ui() override; // Draws into the currently active imgui window
    virtual const char* get_display_name() override { return "Smoke"; };

    bool first_frame = true;
    bool one_time_emit = false;

    struct Context* ctx = nullptr;
    VkBuffer shader_globals = VK_NULL_HANDLE;
    GPUBuffer system_globals = {};

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

    glm::vec3 position = glm::vec3(0.0f);
    uint32_t particle_capacity = 0;
    float particle_spawn_rate = 10000.0f;
    //float particle_spawn_rate = 1.0f;
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
    glm::vec3 smoke_dir = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 smoke_origin = glm::vec3(0.0f);

    Texture particle_render_target;
    Texture light_render_target;
    VkSampler light_sampler;

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

struct TrailBlazerSystem : IConfigUI
{
    void init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format);
    void simulate(VkCommandBuffer cmd, float dt);
    void render(VkCommandBuffer cmd);
    void destroy();
    void set_position(glm::vec3 pos) { position = pos; }

    // IConfigUI
    virtual void draw_config_ui() override;
    virtual const char* get_display_name() override;

    struct Context* ctx = nullptr;
    VkBuffer shader_globals = VK_NULL_HANDLE;
    uint32_t particle_capacity = 32678;
    uint32_t child_particle_capacity = 524288;

    float particles_to_spawn = 0.0f;
    float child_particles_to_spawn = 0.0f;
    float time = 0.0f;
    float particle_spawn_rate = 1000.0f;
    float child_spawn_rate = 1000.0f;
    bool first_frame = true;
    bool particles_initialized = false;
    float particle_size = 1.0f;
    float child_size = 0.1f;
    glm::vec4 particle_color = glm::vec4(1.0f);
    float particle_speed = 1.0f;

    struct GraphicsPipelineAsset* render_pipeline = nullptr;
    struct ComputePipelineAsset* particle_emit_pipeline = nullptr;
    struct ComputePipelineAsset* particle_simulate_pipeline = nullptr;

    struct ComputePipelineAsset* child_emit_pipeline = nullptr;
    struct ComputePipelineAsset* child_simulate_pipeline = nullptr;
    struct ComputePipelineAsset* child_dispatch_size_pipeline = nullptr;
    struct ComputePipelineAsset* child_draw_count_pipeline = nullptr;

    // Double buffered
    Buffer particle_buffer[2] = {};
    Buffer particle_system_state[2] = {};

    Buffer indirect_dispatch_buffer = {};
    Buffer indirect_draw_buffer = {};

    Buffer child_particle_buffer[2] = {};
    Buffer child_particle_system_state[2] = {};

    GPUBuffer child_emit_indirect_dispatch_buffer = {};
    Buffer child_indirect_dispatch_buffer = {};
    Buffer child_indirect_draw_buffer = {};

    glm::vec3 position = glm::vec3(0.0f);
};

struct ParticleSystemSimple : IConfigUI
{
    struct Config
    {
        std::string name;
        uint32_t particle_capacity;
        float spawn_rate;
        std::string emit_and_simulate_file;
		std::vector<DescriptorInfo> additional_descriptors;
        bool emit_indirect_dispatch_handled_externally = false;
    };

    void init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format, const Config& cfg);

    void pre_update(VkCommandBuffer cmd, float dt, const GPUBuffer& curr_state, const GPUBuffer& next_state, uint32_t system_index);
    void emit(VkCommandBuffer cmd, float dt);
    void update(VkCommandBuffer cmd, float dt, VkBuffer indirect_dispatch_buffer, uint32_t buffer_offset);
    void post_update(VkCommandBuffer cmd, float dt);

    void render(VkCommandBuffer cmd, VkBuffer indirect_draw_buffer, size_t offset);
    void destroy();

    // ConfigUI
    virtual void draw_config_ui() override;
    virtual const char* get_display_name() override;

    Config config = {};
    
    struct Context* ctx = nullptr;
    VkBuffer shader_globals = VK_NULL_HANDLE;

    bool particles_initialized = false;

    float particles_to_spawn = 0.0f;
    float time = 0.0f;

    bool emit_indirect_dispatch_handled_externally = false;

    struct GraphicsPipelineAsset* render_pipeline = nullptr;
    struct ComputePipelineAsset* particle_emit_pipeline = nullptr;
    struct ComputePipelineAsset* particle_simulate_pipeline = nullptr;

    // Double buffered
    Buffer particle_buffer[2] = {};

    Buffer emit_indirect_dispatch_buffer = {};

    std::vector<DescriptorInfo> descriptors;
    ParticleTemplatePushConstants push_constants;
};

struct ParticleManagerSimple
{
    std::vector<ParticleSystemSimple*> systems;
    Context* ctx;
    VkBuffer globals_buffer;
	VkFormat render_target_format;
    bool initialized = false;
    ComputePipelineAsset* write_indirect_dispatch = nullptr;
    ComputePipelineAsset* write_indirect_draw = nullptr;
    static constexpr uint32_t MAX_SYSTEMS = 1024;
    GPUBuffer system_states_buffer[2]; // Double buffered
    Buffer indirect_dispatch_buffer;
    Buffer indirect_draw_buffer;
    bool first_frame = true;

	void init(Context* ctx, VkBuffer globals_buffer, VkFormat render_target_format);
    ParticleSystemSimple* add_system(const ParticleSystemSimple::Config& cfg);
    void update_systems(VkCommandBuffer cmd, float dt);
    void render_systems(VkCommandBuffer cmd);
    void destroy();
};