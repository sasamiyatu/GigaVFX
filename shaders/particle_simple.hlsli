#include "particle_emitters.hlsli"
#include "noise.hlsli"
#include "color.hlsli"

bool particle_init(uint3 thread_id, inout GPUParticle p, float delta_time, uint4 seed)
{
    float3 sphere_pos = emit_uniform_sphere(seed);
    float3 color = normalize(sphere_pos) * 0.5 + 0.5;
    p.position = float3(0, 1, 2) + sphere_pos * 0.2;
    //float3 dir = float3(sin(globals.time), 0.0f, cos(globals.time));
    float3 dir = float3(0, 1, 0);
    float3 random_dir = emit_uniform_sphere(seed);
    float speed = uniform_random(seed).x * 3.0 + 4.0;

    dir = lerp(random_dir, dir, 0.85);

    p.velocity = dir * speed;
    p.position += uniform_random(seed).x * delta_time * p.velocity;
    p.lifetime = p.max_lifetime = uniform_random(seed).x * 5.4 + 0.8;
    p.size = 0.1;
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
    const float drag = 0.5;
    const float restitution = 0.5;
    const float friction = 0.5;
    float3 gravitational_force = float3(0, -9.8, 0);
    float3 wind_velocity = float3(-5, 0, 0);
    float3 air_resistance = drag * (wind_velocity - p.velocity);

    float height_before = p.position.y;
    float3 acceleration = air_resistance + gravitational_force;
    p.velocity += acceleration * delta_time;
    p.position += p.velocity * delta_time;
    float height_after = p.position.y;


    if (sign(height_before) != sign(height_after))
    {
        float3 n = float3(0, 1, 0);
        float3 vn = dot(p.velocity, n) * n;
        float3 vt = p.velocity - vn;
        p.position -= height_after * (1 + restitution);
        p.velocity = -restitution * vn + (1 - friction) * vt;
    }
    p.size = p.lifetime / p.max_lifetime * 0.1;
    p.lifetime -= delta_time;

    return p.lifetime > 0;
}

bool particle_shade(GPUParticle p, float2 uv, out float4 color)
{
    float dist = distance(uv, float2(0.5, 0.5));
    if (dist > 0.5) return false;

    float t = 1 - p.lifetime / p.max_lifetime;
    float3 c = iq_palette(t,
        float3(0.5, 0.5, 0.5),
        float3(0.5, 0.5, 0.5),
        float3(1.0, 1.0, 1.0),
        float3(0.00, 0.10, 0.20));

    float alpha = 1.0  - smoothstep(0.5, 1.0, t);

    color = float4(c * alpha, alpha);

    return true;
}