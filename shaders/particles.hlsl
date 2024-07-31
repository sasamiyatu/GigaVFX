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
    float2 uv: TEXCOORD0;
};

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] SamplerState texture_sampler;
[[vk::binding(2)]] Texture2D texture;

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
    float2 uv = offsets[indices[input.vertex_id]] * 0.5f + 0.5f;
    float2 uv_scale = float2(1.0 / float2(push_constants.flipbook_size));
    float2 uv_offset = float2(uv_scale.x * (push_constants.flipbook_index % push_constants.flipbook_size.x), uv_scale.y * (push_constants.flipbook_index / push_constants.flipbook_size.y));
    output.uv = uv * uv_scale + uv_offset;
    return output;
}

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    float4 tex = texture.Sample(texture_sampler, input.uv);
    tex.rgb = srgb_to_linear(tex.rgb);

    output.color = tex * float4(input.color.rgb, push_constants.normalized_lifetime);

    return output;
}