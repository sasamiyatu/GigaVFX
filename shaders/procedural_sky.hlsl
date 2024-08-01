#include "shared.h"
#include "atmosphere.hlsli"
#include "misc.hlsli"

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] RWTexture2D<float4> out_texture;


struct Ray
{
    float3 origin;
    float3 direction;
};

Ray get_camera_ray(float2 uv, float4x4 view_inverse, float4x4 projection)
{
    view_inverse = transpose(view_inverse);
    projection = transpose(projection);
    uv = uv * 2.0 - 1.0;
    float3 ro = view_inverse[3].xyz;
    float aspect = projection[1][1] / projection[0][0];
    float tan_half_fov_y = 1.f / projection[1][1];
    float3 rd = normalize(
        (uv.x * view_inverse[0].xyz * tan_half_fov_y * aspect) - 
        (uv.y * view_inverse[1].xyz * tan_half_fov_y) -
        view_inverse[2].xyz);

    Ray ray;
    ray.origin = ro;
    ray.direction = rd;
    return ray;
}



[numthreads(8, 8, 1)]
void cs_main( uint3 thread_id : SV_DispatchThreadID )
{
    uint w, h;
    out_texture.GetDimensions(w, h);
    float2 uv = float2(thread_id.xy + 0.5) / float2(w, h);
    if (all(saturate(uv) == uv))
    {
        Ray ray = get_camera_ray(uv, globals.view_inverse, globals.projection);

        float rl = INFINITY;
        float3 sun_radiance = 1.0;
        float fog_factor = 0.0;
        float4 transmittance = 1.0;
        float atmosphere_occlusion = 1.0;
        float3 sun_dir = normalize(globals.sun_direction.rgb);
        float3 scattering = GetAtmosphere(ray.origin, ray.direction, rl, sun_dir, sun_radiance, transmittance, fog_factor) * atmosphere_occlusion;

        float3 color = scattering;
        color += GetSunDisc(ray.direction, sun_dir) * sun_radiance * 1000.0 * transmittance.w;

        out_texture[thread_id.xy] = float4(color, 1.0);
    }
}