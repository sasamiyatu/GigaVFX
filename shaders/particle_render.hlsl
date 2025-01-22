#include "shared.h"

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] StructuredBuffer<GPUParticle> particles;

bool particle_shade(in GPUParticle p, float2 uv, out float4 color);

struct VSInput
{
    uint vertex_id: SV_VertexID;
    uint instance_id: SV_InstanceID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float4 color: COLOR0;
    uint particle_index: TEXCOORD0;
    [[vk::builtin("PointSize")]] float point_size : PSIZE;
};

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;

    GPUParticle p = particles[input.instance_id];
    float4 pos = float4(p.position, 1.0);
    float4 view_pos = mul(globals.view, pos);
    output.position = mul(globals.projection, view_pos);

    const float particle_size = p.size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(globals.projection, corner);
    float point_size = globals.resolution.x * proj_corner.x / proj_corner.w;

    output.color = p.color;
    output.particle_index = input.instance_id;

    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    return output;
}

struct PSInput
{
    float4 position: SV_Position;
    float4 color: COLOR0;
    uint particle_index: TEXCOORD0;
};

struct PSOutput
{
    float4 color: SV_Target0;
};

[[vk::ext_builtin_input(/* PointCoord */ 16)]]
static const float2 gl_PointCoord;


PSOutput fs_main(PSInput input)
{
    PSOutput output = (PSOutput)0;

    GPUParticle p = particles[input.particle_index];
    float4 col;
    if (!particle_shade(p, gl_PointCoord.xy, col))
        discard;

    output.color = col;

    return output;
}