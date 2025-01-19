#include "pbr.hlsli"
#include "shared.h"
#include "math.hlsli"
#include "misc.hlsli"
#include "pcss.hlsli"

#define BINDLESS_DESCRIPTOR_SET_INDEX 1

struct VSInput
{
    uint vertex_id: SV_VertexID;
    uint instance_id: SV_InstanceID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float2 texcoord0: TEXCOORD0;
    float3 normal: NORMAL;
    float4 tangent: TANGENT;
    float3 world_position: POSITION;
    float3 view_position: TEXCOORD1;
};

[[vk::constant_id(1)]] const bool USE_INSTANCING = false;

[[vk::binding(0, BINDLESS_DESCRIPTOR_SET_INDEX)]] Texture2D<float4> bindless_textures[];

[[vk::binding(0)]] SamplerState bilinear_sampler;
[[vk::binding(1)]] cbuffer globals {
    ShaderGlobals globals;
}
[[vk::binding(2)]] StructuredBuffer<Material> materials;
[[vk::binding(3)]] Texture2DArray shadowmap_texture;
[[vk::binding(4)]] SamplerComparisonState shadow_sampler;
[[vk::binding(5)]] SamplerState point_sampler;
[[vk::binding(6)]] Texture2D particle_light_texture;
//[[vk::binding(7)]] StructuredBuffer<float4x4> transforms;

[[vk::push_constant]]
PushConstantsForward push_constants;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 pos = vk::RawBufferLoad<float3>(push_constants.position_buffer + input.vertex_id * 12);
    //float4x4 model = USE_INSTANCING ? transforms[input.instance_id] : push_constants.model;
    float4x4 model = push_constants.model;
    float3 world_pos = mul(model, float4(pos, 1.0)).xyz;
    float3 normal = push_constants.normal_buffer ? vk::RawBufferLoad<float3>(push_constants.normal_buffer + input.vertex_id * 12) : float3(0.0, 0.0, 1.0);
    
    output.view_position = mul(globals.view, float4(world_pos, 1.0)).xyz;
    output.position = mul(globals.viewprojection, float4(world_pos, 1.0));
    output.world_position = world_pos;
    output.normal  = mul(model, float4(normal, 0.0)).xyz;

    if (push_constants.texcoord0_buffer)    output.texcoord0 = vk::RawBufferLoad<float2>(push_constants.texcoord0_buffer + input.vertex_id * 8);
    if (push_constants.tangent_buffer)      output.tangent = vk::RawBufferLoad<float4>(push_constants.tangent_buffer + input.vertex_id * 16);
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
    //if (material_context.basecolor.a < material.alpha_cutoff) discard;

    float3 V = normalize(globals.camera_pos.xyz - input.world_position);
    float3 N = material_context.shading_normal;
    float3 L = normalize(globals.sun_direction.xyz);

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

    // Particles can cast colored shadows
    float3 shadow = 1.0;

    const float3 cascade_colors[4] = {float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0), float3(1.0, 0.0, 1.0)};
    
    float view_z = abs(input.view_position.z);
    float cascade = dot(globals.shadow_cascade_thresholds < view_z, 1.0) - 1.0;

    { // Shadow map
        float4 shadow_space = mul(globals.shadow_view_projection[cascade], float4(input.world_position, 1.0));
        shadow_space.xyz /= shadow_space.w;
        float2 shadow_uv = shadow_space.xy * 0.5 + 0.5;
        shadow_uv.y = 1.0 - shadow_uv.y;
        float bias = 0.005 / shadow_space.w;
        float compare_value = shadow_space.z - bias;
        if (all(saturate(shadow_uv) == shadow_uv))
        {
            float w, h, elements;
            shadowmap_texture.GetDimensions(w, h, elements);
#if 0
            float4 gathered = shadowmap_texture.Gather(point_sampler, shadow_uv).wzxy;
            float2 pixel = shadow_uv * float2(w, h);
            pixel -= 0.5;
            float2 fracts = frac(pixel);
            float4 weights = float4((1.0 - fracts.y) * (1.0 - fracts.x), (1.0 - fracts.y) * fracts.x, fracts.y * (1.0 - fracts.x), fracts.y * fracts.x);
            shadow = dot(gathered > compare_value, weights);
            shadow = PCSS(shadowmap_texture, point_sampler, shadow_sampler, globals.shadow_projection_info[0], float4(shadow_uv, compare_value, 1.0));
#else
            float shd = 0.0;
            float2 pixel_size = float2(1.0 / w, 1.0 / h);
            for (int i = 0; i < PCF_NUM_SAMPLES; ++i)
            {
                float s = shadowmap_texture.SampleCmp(shadow_sampler, float3(shadow_uv + poissonDisk[i] * pixel_size * 2.0, cascade), compare_value).r;
                shd += s;
            }

            shd /= float(PCF_NUM_SAMPLES);
            shadow = shd;
#endif
        }

        if (any(shadow > 0.0))
        { // Particle shadows
            uint cascade = 1; // Particles use first cascade only
            float4 shadow_space = mul(globals.shadow_view_projection[cascade], float4(input.world_position, 1.0));
            shadow_space.xyz /= shadow_space.w;
            float2 shadow_uv = shadow_space.xy * 0.5 + 0.5;
            shadow_uv.y = 1.0 - shadow_uv.y;

            float4 light = particle_light_texture.Sample(bilinear_sampler, shadow_uv);
            shadow *= 1.0 - light.rgb;
        }
    }

    float3 ambient = material_context.basecolor.rgb * 0.04;
    float3 radiance = (Fr + Fd) * NdotL * shadow + ambient;
    //radiance = rgb_to_luminance(radiance) * cascade_colors[int(cascade)];
    output.color = float4(radiance, 1.0);
    //output.color.rgb = N * 0.5 + 0.5;

    return output;
}