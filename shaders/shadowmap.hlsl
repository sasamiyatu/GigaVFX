#include "pbr.hlsli"
#include "shared.h"
#include "math.hlsli"
#include "misc.hlsli"

[[vk::constant_id(1)]] const bool CAN_DISINTEGRATE = false;

#define BINDLESS_DESCRIPTOR_SET_INDEX 1

struct VSInput
{
    uint vertex_id: SV_VertexID;
    uint view_id: SV_ViewID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float2 texcoord0: TEXCOORD0;
};

[[vk::binding(0, BINDLESS_DESCRIPTOR_SET_INDEX)]] Texture2D<float4> bindless_textures[];

[[vk::binding(0)]] SamplerState bilinear_sampler;
[[vk::binding(1)]] cbuffer globals {
    ShaderGlobals globals;
}
[[vk::binding(2)]] StructuredBuffer<Material> materials;

[[vk::push_constant]]
DepthPrepassPushConstants push_constants;
//PushConstantsForward push_constants;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 pos = vk::RawBufferLoad<float3>(push_constants.position_buffer + input.vertex_id * 12);
    float3 world_pos = mul(push_constants.model, float4(pos, 1.0)).xyz;
    output.position = mul(globals.shadow_view_projection[input.view_id], float4(world_pos, 1.0));
    if (push_constants.texcoord0_buffer)
        output.texcoord0 = vk::RawBufferLoad<float2>(push_constants.texcoord0_buffer + input.vertex_id * 8);
    return output;
}

void fs_main(VSOutput input)
{
    if (CAN_DISINTEGRATE)
    {
        float alpha = bindless_textures[push_constants.noise_texture_index].Sample(bilinear_sampler, input.texcoord0).r;
        alpha = srgb_to_linear(alpha.xxx).x;
        if (alpha < push_constants.alpha_reference)
            discard;
    }
}