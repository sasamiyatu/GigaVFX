#include "particle_emitters.hlsli"
#include "noise.hlsli"
#include "color.hlsli"

bool particle_init(uint3 thread_id, inout GPUParticle p, float delta_time, uint4 seed)
{
    float3 sphere_pos = emit_uniform_sphere(seed);

    float theta = uniform_random(seed).x * TWO_PI;
    p.position = float3(0, 0, -2) + float3(cos(theta), 0, sin(theta)) * 1;
    //float3 dir = float3(sin(globals.time), 0.0f, cos(globals.time));
    float3 dir = float3(0, 1, 0);
    float3 random_dir = emit_uniform_sphere(seed);
    float speed = uniform_random(seed).x * 3.0 + 4.0;

    dir = lerp(random_dir, dir, 1);

    p.velocity = dir * speed;
    p.position += uniform_random(seed).x * delta_time * p.velocity;
    p.lifetime = p.max_lifetime = uniform_random(seed).x * 0.2 + 0.4;
    p.size = 0.02;
    //p.color = float4(uniform_random(seed).rgb, 1.0);

    return true;
}

float sdf_sphere(float3 p, float3 ce, float r)
{
    return length(p - ce) - r;
}

float sdf_func(float3 p)
{
    return (sdf_sphere(p, float3(0, 1, -5), 0.2));
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
    float4 phi = gradient_noise_deriv(p.position);
    float4 psi = gradient_noise_deriv((p.position +  float3(31.416, -47.853, 12.679)));

    float3 n = sdf_normal(p.position);
    float d = sdf_func(p.position);
    float3 v = lerp(n, phi, max(0.1, smoothstep(0.0, 1.0, abs(d))));
    float3 crossed_grad = cross(psi.yzw, phi.yzw);

    p.velocity = float3(0, 1, 0) + crossed_grad;
    p.position += p.velocity * delta_time;
    p.lifetime -= delta_time;
    p.size = p.lifetime / p.max_lifetime * 0.02;

    return p.lifetime > 0;
}

bool particle_shade(GPUParticle p, float2 uv, out float4 color)
{
    float dist = distance(uv, float2(0.5, 0.5));
    if (dist > 0.5) return false;

    float age = 1 - p.lifetime / p.max_lifetime;

    float brightness = 1 - smoothstep(0.0, 1.0, age);
    float3 col = float3(2.4, 0.8, 0.3) * brightness * 3;
    col = lerp(col, float3(0, 0, 0), age);
    float alpha = 1 - smoothstep(0.5, 1.0, age);
    color = float4(col * alpha, alpha);

    return true;
}