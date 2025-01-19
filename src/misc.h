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

inline uint32_t get_golden_dispatch_size(uint32_t size)
{
    constexpr uint32_t golden_workgroup_size = 8;
    return align_power_of_2(size, golden_workgroup_size) / golden_workgroup_size;
}

inline uint8_t* read_entire_file(const char* filepath, size_t* size)
{
    assert(size);
    FILE* f = fopen(filepath, "rb");
    if (!f) 
    {
        LOG_ERROR("Failed to open file %s", filepath);
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(filesize);
    assert(data);
    size_t bytes_read = fread(data, 1, filesize, f);
    assert(bytes_read == filesize);
    *size = bytes_read;

    return data;
}

inline std::string read_text_file(const char* filepath)
{
    FILE* f = fopen(filepath, "rb");
    if (!f)
    {
        LOG_ERROR("Failed to open file %s", filepath);
        return std::string();
    }

    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(filesize);
    assert(data);
    size_t bytes_read = fread(data, 1, filesize, f);
    assert(bytes_read == filesize);

    return std::string((const char*)data, (size_t)filesize);
}