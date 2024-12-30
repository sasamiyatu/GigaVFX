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
    float4x4 viewprojection;
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
    uint64_t blas_address;
    uint particles_to_spawn;
    float particle_size;
    uint num_slices;
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
    uint active_particle_count;
};

struct GPUParticle
{
    float3 position;
    float3 velocity;
    float lifetime; // alive if > 0
};

struct GPUParticleSort
{
    uint index;
    uint key;
};
