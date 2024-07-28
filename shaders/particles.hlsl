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
    float4 color: COLOR0;
};

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

static const float2 offsets[4] = {
    float2(-1.0f, 1.0f),
    float2(-1.0f, -1.0f),
    float2(1.0f, -1.0f),
    float2(1.0f, 1.0f)
};

static const uint indices[6] = {
    0,
    1,
    2,
    2,
    3,
    0
};

[[vk::push_constant]]
PushCostantsParticles push_constants;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 view_pos = mul(globals.view, float4(push_constants.position.xyz, 1.0f)).xyz;
    view_pos.xy += offsets[indices[input.vertex_id]] * push_constants.size;
    output.position = mul(globals.projection, float4(view_pos, 1.0));
    output.color = push_constants.color;
    return output;
}

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    output.color = float4(input.color.rgb, push_constants.normalized_lifetime);

    return output;
}