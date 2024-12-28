#pragma once

float3 linear_to_srgb(float3 color)
{
    float3 cutoff = color < 0.0031308;
    float3 higher = 1.055*pow(color, 1.0/2.4) - 0.055;
	float3 lower = color * 12.92;
    return select(cutoff, lower, higher);
}

float3 srgb_to_linear(float3 color)
{
    float3 cutoff = color < 0.04045;
	float3 higher = pow((color + 0.055)/1.055, 2.4);
	float3 lower = color/12.92;

    return select(cutoff, lower, higher);
}

float rgb_to_luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

uint sort_key_from_float(uint f)
{
	uint mask = -int(f >> 31) | 0x80000000;
	return f ^ mask;
}