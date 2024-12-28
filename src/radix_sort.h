#pragma once

#include "graphics_context.h"
#include "radix_sort/radix_sort_vk.h"

struct RadixSortContext;
RadixSortContext* radix_sort_context_create(Context* ctx, uint32_t max_count);
void radix_sort_context_destroy(RadixSortContext* ctx);
void test_radix_sort(Context* ctx);
