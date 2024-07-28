#include "pbr.hlsli"
#include "shared.h"
#include "math.hlsli"
#include "misc.hlsli"

struct VSInput
{
    uint vertex_id: SV_VertexID;

};

struct VSOutput
{
    float4 position: SV_Position;
    [[vk::builtin("PointSize")]]
    float point_size : PSIZE;
    float4 color: COLOR0;
};

struct FSInput
{
    float4 position: SV_Position;
    float4 color: COLOR0;
};

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::push_constant]]
PushCostantsParticles push_constants;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 world_pos = push_constants.position.xyz;
    output.position = mul(globals.viewprojection, float4(world_pos, 1.0));
    output.point_size = 10.0f;
    return output;
}

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(FSInput input)
{
    PSOutput output = (PSOutput)0;
    output.color = input.color;

    return output;
}