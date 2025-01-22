#pragma once

#include "math.hlsli"

// https://www.pcg-random.org/
uint pcg(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

uint2 pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v>>16u);

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v>>16u);

    return v;
}

// http://www.jcgt.org/published/0009/03/02/
uint3 pcg3d(uint3 v) {

    v = v * 1664525u + 1013904223u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v ^= v >> 16u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    return v;
}

// http://www.jcgt.org/published/0009/03/02/
uint4 pcg4d(inout uint4 seed)
{
    seed = seed * 1664525u + 1013904223u;
    seed += seed.yzxy * seed.wxyz;
    seed = (seed >> 16) ^ seed;
    seed += seed.yzxy * seed.wxyz;
    return seed;
}

float4 uniform_random(inout uint4 seed)
{
    return float4(pcg4d(seed)) / float(0xFFFFFFFFu);
}

float4 sample_uniform(inout uint4 seed, float range_min, float range_max)
{
    return uniform_random(seed) * (range_max - range_min) + range_min;
}

float4 sample_gaussian(inout uint4 seed, float mean = 0, float std_dev = 1)
{
    // Box-Muller transform
    float4 r = uniform_random(seed);

    float2 s1 = float2(sqrt(-2 * log(r.x)), TWO_PI * r.y);
    float2 s2 = float2(sqrt(-2 * log(r.z)), TWO_PI * r.w);

    float4 gaussian;
    gaussian.x = s1.x * cos(s1.y);
    gaussian.y = s1.x * sin(s1.y);
    gaussian.z = s2.x * cos(s2.y);
    gaussian.w = s2.x * sin(s2.y);

    return gaussian * std_dev + mean;
}