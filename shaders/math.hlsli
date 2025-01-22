#pragma once

#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define ONE_OVER_SQRT_3 0.57735026919

float3 sample_uniform_sphere(float2 xi)
{
    // Compute radius r (branchless).
    xi = 2.0 * xi - 1.0;
    float d = 1.0 - (abs(xi.x) + abs(xi.y));
    float r = 1.0 - abs(d);
    // Compute phi in the first quadrant (branchless, except for the
    // division-by-zero test), using sign(u) to map the result to the
    // correct quadrant below.
    float phi = (r == 0.0) ? 0.0 : (PI/4.0) * ((abs(xi.y) - abs(xi.x)) / r + 1.0);
    float f = r * sqrt(2.0 - r*r);
    float x = f * sign(xi.x) * cos(phi);
    float y = f * sign(xi.y) * sin(phi);
    float z = sign(d) * (1.0 - r*r);
    float pdf = 1.0 / (4.0*PI);

    return float3(x, y, z);
}

float3x3 create_tangent_space(float3 normal)
{
    float3 major;
    if(abs(normal.x) < ONE_OVER_SQRT_3) major = float3(1,0,0);
    else if(abs(normal.y) < ONE_OVER_SQRT_3) major = float3(0,1,0);
    else major = float3(0,0,1);

    float3 tangent = normalize(cross(normal, major));

    float3 bitangent = cross(normal, tangent);
    return float3x3(tangent, bitangent, normal);
}