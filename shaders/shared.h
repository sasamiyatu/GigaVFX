#pragma once

#if __cplusplus

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtx/compatibility.hpp"

using namespace glm;

#define FUNC_QUALIFIER inline

#else

#define alignas(x) 
#define FUNC_QUALIFIER

#endif

FUNC_QUALIFIER float linearize_depth(float depth, float4 projection_info)
{
    return depth * projection_info.x + projection_info.y;
}

struct alignas(16) Material
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
    float4 camera_pos;
    float4 sun_direction;
    float4 sun_color_and_intensity;
};