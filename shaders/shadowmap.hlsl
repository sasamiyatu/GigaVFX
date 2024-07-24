#include "pbr.hlsli"
#include "shared.h"
#include "math.hlsli"
#include "misc.hlsli"

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
PushConstantsForward push_constants;

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

struct PSOutput
{
    float4 color: SV_Target0;
};

struct MaterialContext
{
    float4 basecolor;
    float3 smooth_normal;
    float3 shading_normal;
    float3 f0;
    float metallic;
    float roughness;
};

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    //float4 basecolor = basecolor_texture.Sample(bilinear_sampler, input.texcoord0);
    //Material material = materials.Load(push_constants.material_index);
    //MaterialContext material_context = load_material_context(input, material);
    //if (material_context.basecolor.a < material.alpha_cutoff)
    //    discard;

    return output;
}