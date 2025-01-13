#pragma once

#include "defines.h"
#include <vector>
#include "buffer.h"

struct Mesh
{
    struct Primitive
    {
        int32_t material; // -1 if default material
        uint32_t first_vertex;
        uint32_t first_index;
        uint32_t index_count;
    };

    std::vector<Primitive> primitives;

    Buffer indices;
    Buffer position;
    Buffer normal;
    Buffer tangent;
    Buffer texcoord0;
    Buffer texcoord1;
};