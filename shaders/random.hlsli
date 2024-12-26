#pragma once

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
