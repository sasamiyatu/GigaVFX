#pragma once

#include "graphics_context.h"
#include "radix_sort/radix_sort_vk.h"

struct RadixSortContext;
RadixSortContext* radix_sort_context_create(Context* ctx, uint32_t max_count);
void radix_sort_context_destroy(RadixSortContext* ctx);
const Buffer& radix_sort_context_get_input(RadixSortContext* ctx);
void radix_sort_sort_direct(RadixSortContext* ctx, uint32_t count, VkCommandBuffer cmd, const Buffer** output_buffer);
void radix_sort_sort_indirect(RadixSortContext* ctx, uint32_t count, VkCommandBuffer cmd, const Buffer** output_buffer);

void test_radix_sort(Context* ctx);
