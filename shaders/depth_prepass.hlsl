#include "shared.h"
#include "misc.hlsli"

#define BINDLESS_DESCRIPTOR_SET_INDEX 1

[[vk::constant_id(1)]] const bool CAN_DISINTEGRATE = false;

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] SamplerState bilinear_sampler;

[[vk::binding(2)]] RWStructuredBuffer<DispatchIndirectCommand> particle_count;
[[vk::binding(3)]] RWStructuredBuffer<float2> particles_spawned;

[[vk::binding(0, BINDLESS_DESCRIPTOR_SET_INDEX)]] Texture2D<float4> bindless_textures[];

struct VSInput
{
    uint vertex_id: SV_VertexID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
};

[[vk::push_constant]]
DepthPrepassPushConstants push_constants;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 pos = vk::RawBufferLoad<float3>(push_constants.position_buffer + input.vertex_id * 12);
    float3 world_pos = mul(push_constants.model, float4(pos, 1.0)).xyz;
    
    output.position = mul(globals.viewprojection, float4(world_pos, 1.0));
    
    if (CAN_DISINTEGRATE)
    {
        output.uv = vk::RawBufferLoad<float2>(push_constants.texcoord0_buffer + input.vertex_id * 8);
    }

    return output;
}

void fs_main(VSOutput input)
{
    if (CAN_DISINTEGRATE)
    {
        float alpha = bindless_textures[push_constants.noise_texture_index].Sample(bilinear_sampler, input.uv).r;
        alpha = srgb_to_linear(alpha.xxx).x;

        if (alpha < push_constants.alpha_reference)
        {
            if (alpha > push_constants.prev_alpha_reference)
            {
                uint index = 0;
                InterlockedAdd(particle_count[0].x, 1, index);
                printf("idx: %d", index);
                particles_spawned[index] = input.position.xy / globals.resolution.xy;
            }

            discard;
        }
    }
}