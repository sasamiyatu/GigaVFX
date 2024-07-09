#include "pbr.hlsli"
#include "shared.h"
#include "math.hlsli"

#define BINDLESS_DESCRIPTOR_SET_INDEX 1

struct VSInput
{
    uint vertex_id: SV_VertexID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float2 texcoord0: TEXCOORD0;
    float3 normal: NORMAL;
    float4 tangent: TANGENT;
    float3 world_position: POSITION;
};

[[vk::binding(0, BINDLESS_DESCRIPTOR_SET_INDEX)]] Texture2D<float4> bindless_textures[];

[[vk::binding(0)]] SamplerState bilinear_sampler;
[[vk::binding(1)]] cbuffer globals {
    float4x4 view;
    float4x4 projection;
    float4x4 viewprojection;
    float4 camera_pos;
    float4 sun_direction;
    float4 sun_color_and_intensity;
}
[[vk::binding(2)]] StructuredBuffer<Material> materials;

[[vk::push_constant]]
PushConstantsForward push_constants;

float3 linear_to_srgb(float3 color)
{
    float3 cutoff = color < 0.0031308;
    float3 higher = 1.055*pow(color, 1.0/2.4) - 0.055;
	float3 lower = color * 12.92;
    return select(cutoff, lower, higher);
}

float3 srgb_to_linear(float3 color)
{
    float3 cutoff = color < 0.04045;
	float3 higher = pow((color + 0.055)/1.055, 2.4);
	float3 lower = color/12.92;

    return select(cutoff, lower, higher);
}

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 pos = vk::RawBufferLoad<float3>(push_constants.position_buffer + input.vertex_id * 12);
    float3 world_pos = mul(push_constants.model, float4(pos, 1.0)).xyz;
    output.world_position = world_pos;
    output.position = mul(viewprojection, float4(world_pos, 1.0));
    if (push_constants.normal_buffer)
        output.normal = vk::RawBufferLoad<float3>(push_constants.normal_buffer + input.vertex_id * 12);
    if (push_constants.texcoord0_buffer)
        output.texcoord0 = vk::RawBufferLoad<float2>(push_constants.texcoord0_buffer + input.vertex_id * 8);
    if (push_constants.tangent_buffer)
        output.tangent = vk::RawBufferLoad<float4>(push_constants.tangent_buffer + input.vertex_id * 16);
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

MaterialContext load_material_context(VSOutput input, Material material)
{
    MaterialContext ctx;
    ctx.basecolor = material.basecolor_factor;
    if (material.basecolor_texture != -1)
    {
        float4 tex = bindless_textures[material.basecolor_texture].Sample(bilinear_sampler, input.texcoord0);
        tex.rgb = srgb_to_linear(tex.rgb);
        ctx.basecolor *= tex;
    }

    ctx.smooth_normal = normalize(input.normal);
    if (material.normal_texture != -1)
    {
        float3 normal_map = bindless_textures[material.normal_texture].Sample(bilinear_sampler, input.texcoord0).rgb;
        float3 geometric_normal = ctx.smooth_normal;
        float3 tangent_space_normal = normalize(normal_map * 2.0 - 1.0);
        float4 tangent = float4(normalize(input.tangent.xyz), input.tangent.w);
        float3 bitangent = cross(tangent.xyz, geometric_normal) * tangent.w;
        float3x3 tbn = float3x3(tangent.xyz, bitangent, geometric_normal);
        float3 world_space_normal = mul(tangent_space_normal, tbn);
        ctx.shading_normal = world_space_normal;
    }
    else
    {
        ctx.shading_normal = ctx.smooth_normal;
    }

    ctx.roughness = material.roughness_factor;
    ctx.metallic = material.metallic_factor;
    if (material.metallic_roughness_texture != -1)
    {
        float2 roughness_metallic = bindless_textures[material.metallic_roughness_texture].Sample(bilinear_sampler, input.texcoord0).gb;
        ctx.roughness *= roughness_metallic.r;
        ctx.metallic *= roughness_metallic.g;
    }

    ctx.roughness *= ctx.roughness;

    ctx.f0 = lerp(DIELECTRIC_F0, ctx.basecolor.rgb, ctx.metallic);

    return ctx;
}

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    //float4 basecolor = basecolor_texture.Sample(bilinear_sampler, input.texcoord0);
    Material material = materials.Load(push_constants.material_index);
    MaterialContext material_context = load_material_context(input, material);
    if (material_context.basecolor.a < material.alpha_cutoff)
        discard;

    float3 V = normalize(camera_pos.xyz - input.world_position);
    float3 N = material_context.shading_normal;
    float3 L = normalize(sun_direction.xyz);

    float NdotV = abs(dot(N, V)) + 1e-5f;
    float3 H = normalize(V + L);
    float LdotH = saturate(dot(L, H));
    float NdotH = saturate(dot(N, H));
    float NdotL = saturate(dot(N, L));

    // Specular BRDF
    const float f90 = 1.0;
    float3 F = F_schlick(material_context.f0, f90, LdotH);
    float Vis = V_smith_ggx_correlated(NdotV, NdotL, material_context.roughness);
    float D = D_ggx(NdotH, material_context.roughness);
    float3 Fr = D * F * Vis / PI;

    // Diffuse BRDF
    float3 Fd = material_context.basecolor.rgb / PI * (1.0 - F);

    float3 radiance = (Fr + Fd) * NdotL;
    float3 final_output = linear_to_srgb(radiance);
    output.color = float4(final_output, 1.0);

    return output;
}