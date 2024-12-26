#pragma once
#define PI 3.14159265359

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