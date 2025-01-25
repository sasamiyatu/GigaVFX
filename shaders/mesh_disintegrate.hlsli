#include "particle_emitters.hlsli"
#include "color.hlsli"
#include "noise.hlsli"
#include "bitangent_noise.hlsli"

[[vk::binding(7)]] StructuredBuffer<float3> spawn_pos;
[[vk::binding(8)]] Texture2D depth_texture;
[[vk::binding(9)]] SamplerState depth_sampler;

bool particle_init(uint3 thread_id, inout GPUParticle p, float delta_time, uint4 seed)
{
    float3 wpos = spawn_pos[thread_id.x];
    float3 sphere_pos = emit_uniform_sphere(seed);
    float3 color = normalize(sphere_pos) * 0.5 + 0.5;
    p.position = wpos;
    //float3 dir = float3(sin(globals.time), 0.0f, cos(globals.time));
    float3 dir = float3(0, 1, 0);
    float3 random_dir = emit_uniform_sphere(seed);
    float speed = uniform_random(seed).x * 3.0 + 4.0;

    dir = lerp(random_dir, dir, 0.85);

    //p.velocity = dir * speed;
    p.position += uniform_random(seed).x * delta_time * p.velocity;
    p.lifetime = p.max_lifetime = uniform_random(seed.y) * 0.5 + 0.5;
    p.size = 0.005;
    //p.color = float4(uniform_random(seed).rgb, 1.0);

    return true;
}

float sdf_sphere(float3 p, float3 ce, float r)
{
    return length(p - ce) - r;
}

float sdf_func(float3 p)
{
    return (sdf_sphere(p, float3(0, 1, 4), 1.0));
}

float3 sdf_normal( in float3 p ) // for function f(p)
{
    const float eps = 0.001; // or some other value
    const float2 h = float2(eps,0);
    return normalize( float3(sdf_func(p+h.xyy) - sdf_func(p-h.xyy),
                           sdf_func(p+h.yxy) - sdf_func(p-h.yxy),
                           sdf_func(p+h.yyx) - sdf_func(p-h.yyx) ) );
}


bool particle_update(uint3 thread_id, inout GPUParticle p, float delta_time, uint4 seed)
{
    float drag = 0.5;
    float3 wind = (float3(0, 3, 0) + BitangentNoise4D(float4(p.position, globals.time)) * 0.4);
    float3 force = drag * (wind - p.velocity);

    p.velocity += force * delta_time;
    p.position += p.velocity * delta_time;

    p.lifetime -= delta_time;

    return p.lifetime > 0;
}

bool particle_shade(GPUParticle p, float2 uv, out float4 color)
{
    float dist = distance(uv, float2(0.5, 0.5));
    if (dist > 0.5) return false;

    float t = 1 - p.lifetime / p.max_lifetime;
    // float3 c = iq_palette(t,
    //     float3(0.5, 0.5, 0.5),
    //     float3(0.5, 0.5, 0.5),
    //     float3(1.0, 1.0, 1.0),
    //     float3(0.00, 0.10, 0.20));

    float3 c = iq_palette(t, float3(0.8,0.4,0.1),float3(0.5,0.5,0.2),float3(0.5,0.6,0.5),float3(0.0,0.0,0.0) );
    float3 brightness = 1 - smoothstep(0.0, 1.0, t);
    c *= brightness;

    float alpha = 1.0  - smoothstep(0.25, 1.0, t);

    color = float4(c * alpha, alpha);

    return true;
}