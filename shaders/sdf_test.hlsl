#include "shared.h"

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}
[[vk::binding(1)]] SamplerState sdf_sampler;
[[vk::binding(2)]] Texture3D<float> sdf_texture;
[[vk::binding(3)]] RWTexture2D<float4> out_texture;

[[vk::push_constant]]
SDFPushConstants push_constants;

float2 sphere_intersect(in float3 ro, in float3 rd, float3 center, float radius)
{
    float3 oc = ro - center;
    
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, oc);
    float c = dot(oc, oc) - radius * radius;
    
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) return float2(-1, -1);
    
    float d = sqrt(discriminant);
    
    float a2 = 2.0 * a;
    
    return float2((-b - d) / a2, (-b + d) / a2);
}

float3 create_camera_ray(float2 uv, float4x4 proj, float4x4 v_inv)
{
    float aspect = proj[1][1] / proj[0][0];
    float tan_half_fov_y = 1.f / proj[1][1];
    float3 rd = normalize(
        (uv.x * v_inv[0].xyz * tan_half_fov_y * aspect) - 
        (uv.y * v_inv[1].xyz * tan_half_fov_y) -
        v_inv[2].xyz);
    return rd;
}

// Calcs intersection and exit distances, and normal at intersection.
// The ray must be in box/object space. If you have multiple boxes all
// aligned to the same axis, you can precompute 1/rd. If you have
// multiple boxes but they are not alligned to each other, use the 
// "Generic" box intersector bellow this one.
float2 boxIntersection( in float3 ro, in float3 rd, in float3 rad, out float3 oN ) 
{
    float3 m = 1.0/rd;
    float3 n = m*ro;
    float3 k = abs(m)*rad;
    float3 t1 = -n - k;
    float3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
	
    if( tN>tF || tF<0.0) return float2(-1.0, -1.0); // no intersection
    
    oN = -sign(rd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);

    return float2( tN, tF );
}

float sdf_func(in float3 p)
{
    float scale = 0.1;
    float3 bb_min = float3(-12.0467, -5, -8.1513) * scale;
    float3 bb_max = float3(12.0467, 14.9399, 8.1513) * scale;
    float3 extent = bb_max - bb_min;
    float3 local_pos = p - bb_min;
    float3 grid_uv = local_pos / extent;
    return sdf_texture.SampleLevel(sdf_sampler, grid_uv, 0) * scale;
}

float3 sdf_normal( in float3 p ) // for function f(p)
{
    const float eps = 0.001; // or some other value
    const float2 h = float2(eps,0);
    return normalize( float3(sdf_func(p+h.xyy) - sdf_func(p-h.xyy),
                           sdf_func(p+h.yxy) - sdf_func(p-h.yxy),
                           sdf_func(p+h.yyx) - sdf_func(p-h.yyx) ) );
}

[numthreads(8, 8, 1)]
void test_sdf( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= globals.resolution.x || thread_id.y >= globals.resolution.y) return;
    float2 uv = float2(thread_id.xy) / float2(globals.resolution);

    uv = uv * 2.0 - 1.0;
    float3 rd = create_camera_ray(uv, transpose(globals.projection), transpose(globals.view_inverse));
    float3 ro = transpose(globals.view_inverse)[3].xyz;


    float scale = 0.1;
    float3 bb_min = float3(-12.0467, -5, -8.1513) * scale;
    float3 bb_max = float3(12.0467, 14.9399, 8.1513) * scale;
    float3 extent = bb_max - bb_min;
    float3 bb_center = (bb_min + bb_max) * 0.5;

    float3 N;
    float2 hit = boxIntersection(ro, rd, extent * 0.5, N);

    if (hit.x > 0.0)
    { // Ray march sdf
        float t = hit.x;
        const int max_steps = 100;

        bool sdf_hit = false;
        for (int i = 0; i < max_steps; ++i)
        {
            //if (t >= hit.y) break;

            float3 p = ro + rd * t;
            float3 local_pos = p - bb_min;

            //float3 grid_uv = local_pos / float3(push_constants.grid_spacing * push_constants.grid_dims);
            float3 grid_uv = local_pos / extent;
            float d = sdf_func(p);
            //d = max(d, grid_step);
            if (d < 1e-3)
            {
                sdf_hit = true;
                break;
            }

            t += d;
        }

        if (sdf_hit)
        {
            float3 p = ro + t * rd;
            float3 n = sdf_normal(p);
            out_texture[thread_id.xy] = float4(n * 0.5 + 0.5, push_constants.grid_dims.x);
        }
    }
}