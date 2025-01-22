#pragma once

#include "random.hlsli"
#include "math.hlsli"

float3 emit_uniform_sphere(inout uint4 seed)
{
    float4 xi = uniform_random(seed);

    float theta = xi.x * TWO_PI - PI;
    float y = xi.y * 2.0 - 1.0;

    float r = sqrt(1 - y * y);

    return float3(r * cos(theta), y, - r * sin(theta));
}

float3 emit_uniform_around_direction(inout uint4 seed, float3 dir, float angle)
{
    float3x3 tbn = create_tangent_space(dir);

    float2 xi = uniform_random(seed).xy;

    float f = xi.x;
    float phi = sqrt(f) * angle;
    float theta = xi.y * TWO_PI - PI;

    float sphi = sin(phi);

    float3 v = float3(
        cos(theta) * sphi,
        sin(theta) * sphi,
        cos(phi)
    );

    return mul(v, tbn);
}

float3 emit_gaussian_around_direction(inout uint4 seed, float3 dir, float angle)
{
    float3x3 tbn = create_tangent_space(dir);

    float2 xi = uniform_random(seed).xy;

    float f = sample_gaussian(seed, 0.0, angle / 3).x;
    float phi = sqrt(f) * angle;
    float theta = xi.y * TWO_PI - PI;

    float sphi = sin(phi);

    float3 v = float3(
        cos(theta) * sphi,
        sin(theta) * sphi,
        cos(phi)
    );

    return mul(v, tbn);
}