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
    float2 uv0: TEXCOORD0;
    float2 uv1: TEXCOORD1;
};

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] cbuffer globals {
    ParticleRenderSettings render_settings;
}

[[vk::binding(2)]] SamplerState texture_sampler;
[[vk::binding(3)]] Texture2D texture;
[[vk::binding(4)]] Texture2D emission_map;

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

float2x2 rot2d(float angle)
{
    float cost = cos(angle);
    float sint = sin(angle);
    return float2x2(cost, sint, -sint, cost);
}

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 view_pos = mul(globals.view, float4(push_constants.position.xyz, 1.0f)).xyz;
    float2 offset = offsets[indices[input.vertex_id]] * push_constants.size;
    offset = mul(rot2d(push_constants.rotation), offset);
    view_pos.xy += offset;
    output.position = mul(globals.projection, float4(view_pos, 1.0));
    output.color = push_constants.color;
    float2 uv = offsets[indices[input.vertex_id]] * 0.5f + 0.5f;
    float2 uv_scale = float2(1.0 / float2(push_constants.flipbook_size));
    float2 uv_offset0 = float2(uv_scale.x * (push_constants.flipbook_index0 % push_constants.flipbook_size.x), uv_scale.y * (push_constants.flipbook_index0 / push_constants.flipbook_size.y));
    float2 uv_offset1 = float2(uv_scale.x * (push_constants.flipbook_index1 % push_constants.flipbook_size.x), uv_scale.y * (push_constants.flipbook_index1 / push_constants.flipbook_size.y));
    output.uv0 = uv * uv_scale + uv_offset0;
    output.uv1 = uv * uv_scale + uv_offset1;
    return output;
}

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    float4 tex0 = texture.Sample(texture_sampler, input.uv0);
    tex0.rgb = srgb_to_linear(tex0.rgb);

    float4 tex1 = texture.Sample(texture_sampler, input.uv1);
    tex1.rgb = srgb_to_linear(tex1.rgb);
    float4 tex = lerp(tex0, tex1, push_constants.flipbook_blend);
    //tex.rgb *= tex.a * input.color.a;

    float4 emission0 = emission_map.Sample(texture_sampler, input.uv0);
    emission0.rgb = srgb_to_linear(emission0.rgb);

    float4 emission1 = emission_map.Sample(texture_sampler, input.uv1);
    emission1.rgb = srgb_to_linear(emission1.rgb);

    float4 emission = lerp(emission0, emission1, push_constants.flipbook_blend);

    output.color = tex * float4(srgb_to_linear(input.color.rgb), 1.0) * float4(srgb_to_linear(render_settings.albedo_multiplier.rgb), render_settings.albedo_multiplier.a);
    output.color.rgb += emission.rgb * srgb_to_linear(render_settings.emission_multiplier.rgb);
    //output.color.a = tex.a;

    return output;
}