#pragma once 
#include "defines.h"
#include <cmath>

inline uint32_t get_mip_count(uint32_t texture_width, uint32_t texture_height)
{
    return (uint32_t)(std::floor(std::log2(std::max(texture_width, texture_height)))) + 1;
}

inline size_t align_power_of_2(size_t size, size_t alignment)
{
    const size_t mask = alignment - 1;
    assert(alignment != 0 && (alignment & mask) == 0);
    return (size + mask) & ~(mask);
}