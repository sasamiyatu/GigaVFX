#pragma once

#if __cplusplus

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtx/compatibility.hpp"

using namespace glm;

#define FUNC_QUALIFIER inline
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;

#else

#define alignas(x) 
#define FUNC_QUALIFIER

#endif

FUNC_QUALIFIER float linearize_depth(float depth, float4 projection_info)
{
    return depth * projection_info.x + projection_info.y;
}

struct Material
{
    float4 basecolor_factor;
    float roughness_factor;
    float metallic_factor;
    float alpha_cutoff;

    int basecolor_texture;
    int metallic_roughness_texture;
    int normal_texture;
};

struct PushConstantsForward
{
    float4x4 model;
    uint64_t position_buffer;
    uint64_t normal_buffer;
    uint64_t tangent_buffer;
    uint64_t texcoord0_buffer;
    uint64_t texcoord1_buffer;
    int material_index;
    float disintegrate_alpha_reference; // For disintegrate effect
};

struct DepthPrepassPushConstants
{
    float4x4 model;
    uint64_t position_buffer;
    uint64_t texcoord0_buffer;
    int noise_texture_index;
    float alpha_reference;
    float prev_alpha_reference;
};

struct PushCostantsParticles
{
    float4 position;
    float4 color;
    uint2 flipbook_size;
    float size;
    float normalized_lifetime;
    uint flipbook_index0;
    uint flipbook_index1;
    float flipbook_blend;
    float rotation;
};

struct PushConstantsTonemap
{
    uint2 size;
};

struct ShaderGlobals
{
    float4x4 view;
    float4x4 view_inverse;
    float4x4 projection;
    float4x4 projection_inverse;
    float4x4 viewprojection;
    float4x4 viewprojection_inverse;
    float4x4 shadow_view[4];
    float4x4 shadow_projection[4];
    float4x4 shadow_view_projection[4];
    float4 shadow_projection_info[4];
    float4 shadow_cascade_thresholds;
    float4 camera_pos;
    float4 sun_direction;
    float4 sun_color_and_intensity;
    float2 resolution;
    uint frame_index;
    float time;
};

struct ParticleRenderSettings
{
    float4 albedo_multiplier;
    float4 emission_multiplier;
};

struct GPUParticlePushConstants
{
    float4 particle_color;
    float3 sort_axis;
    float delta_time;
    uint particles_to_spawn;
    float particle_size;
    uint num_slices;
    float emitter_radius;
    float speed;
    float time;
    float lifetime;
    float noise_scale;
    float noise_time_scale;
    float3 sdf_origin;
    float sdf_grid_spacing;
    uint3 sdf_grid_dims;
    uint particle_capacity;
    uint children_to_emit;
    float3 smoke_dir;
    float3 smoke_origin;
};

struct GPUParticleSystemGlobals
{
    float4x4 transform;
    float4x4 light_view;
    float4x4 light_proj;
    uint2 light_resolution;
    uint particle_capacity;
};

// Matches VkDispatchIndirectCommand
struct DispatchIndirectCommand 
{
    uint x;
    uint y;
    uint z;
};

// Matches VkDrawIndirectCommand 
struct DrawIndirectCommand 
{
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

// Matches VkAabbPositionsKHR 
struct AABBPositions
{
    float min_x;
    float min_y;
    float min_z;
    float max_x;
    float max_y;
    float max_z;
};

// Matches VkAccelerationStructureInstanceKHR
struct AccelerationStructureInstance 
{
    float matrix[3][4];
    uint instanceCustomIndex:24;
    uint mask:8;
    uint instanceShaderBindingTableRecordOffset:24;
    uint flags:8;
    uint64_t accelerationStructureReference;
};

struct GPUParticleSystemState
{
    uint particles_to_emit;
    uint active_particle_count;
};

struct GPUParticle
{
    float3 position;
    float size;
    float3 velocity;
    float lifetime; // alive if > 0
    float4 color;
    float max_lifetime;
};

struct GPUParticleSort
{
    uint index;
    uint key;
};

struct SDFPushConstants
{
    uint3 grid_dims;
    float grid_spacing;
    float3 grid_origin;
};

struct TrailBlazerPushConstants
{
    uint particles_to_spawn;
    uint particle_capacity;
    float delta_time;
    uint3 sdf_dims;
    float sdf_spacing;
    float3 sdf_origin;
};

struct ParticleTemplatePushConstants
{
    uint particles_to_spawn;
    uint particle_capacity;
    float delta_time;
    uint32_t system_index;
};
