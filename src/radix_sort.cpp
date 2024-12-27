#include "radix_sort.h"
#include "radix_sort/radix_sort_vk.h"
#include "buffer.h"
#include "vk_helpers.h"
#include <algorithm>

struct RadixSortContext
{
    Context* ctx = nullptr;
    Buffer even_buffer;
    Buffer odd_buffer;
    Buffer internal_buffer;
    Buffer indirect_buffer;
};

static void test_direct(Context* ctx)
{
    uint32_t test_data[] = { 8,   7,   3,   6,   9,   1,   4,   5,   2,  10, };
    uint32_t count = (uint32_t)std::size(test_data);

    radix_sort_vk_memory_requirements memory_requirements{};
    radix_sort_vk_get_memory_requirements(ctx->radix_sort_instance, count, &memory_requirements);

    BufferDesc desc{};
    desc.size = memory_requirements.keyvals_size;
    desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    Buffer keyvals_buffers[2] = {};
    for (int i = 0; i < 2; ++i)
        keyvals_buffers[i] = ctx->create_buffer(desc, memory_requirements.keyvals_alignment);

    desc.allocation_flags = 0;
    desc.usage_flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.size = memory_requirements.internal_size;
    Buffer internal_buffer = ctx->create_buffer(desc, memory_requirements.internal_alignment);

    {
        void* mapped;
        vmaMapMemory(ctx->allocator, keyvals_buffers[0].allocation, &mapped);
        memcpy(mapped, test_data, sizeof(test_data));
        vmaUnmapMemory(ctx->allocator, keyvals_buffers[0].allocation);
    }

    VkDescriptorBufferInfo keyvals_even{
        keyvals_buffers[0].buffer,
        0,
        VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo keyvals_odd{
        keyvals_buffers[1].buffer,
        0,
        VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo internal{
        internal_buffer.buffer,
        0,
        VK_WHOLE_SIZE
    };
    radix_sort_vk_sort_info_t sort_info{};
    sort_info.key_bits = 32;
    sort_info.count = count;
    sort_info.keyvals_even = keyvals_even;
    sort_info.keyvals_odd = keyvals_odd;
    sort_info.internal = internal;

    VkHelpers::begin_command_buffer(ctx->transfer_command_buffer, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VkDescriptorBufferInfo keyvals_sorted{};
    radix_sort_vk_sort(ctx->radix_sort_instance, &sort_info, ctx->device, ctx->transfer_command_buffer, &keyvals_sorted);

    vkEndCommandBuffer(ctx->transfer_command_buffer);

    VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    info.commandBufferCount = 1;
    info.pCommandBuffers = &ctx->transfer_command_buffer;

    VK_CHECK(vkQueueSubmit(ctx->graphics_queue, 1, &info, VK_NULL_HANDLE));

    VK_CHECK(vkQueueWaitIdle(ctx->graphics_queue));

    VK_CHECK(vkResetCommandPool(ctx->device, ctx->transfer_command_pool, 0));

    {
        void* mapped;
        const Buffer& out_buffer = keyvals_sorted.buffer == keyvals_buffers[0].buffer ? keyvals_buffers[0] : keyvals_buffers[1];
        vmaMapMemory(ctx->allocator, out_buffer.allocation, &mapped);

        printf("vk-radix-sort direct test:\n");
        printf("Original: [ ");
        for (uint32_t i = 0; i < count; ++i)
            printf("%d ", test_data[i]);
        printf("]\n");

        uint32_t* ptr = (uint32_t*)mapped;
        printf("Sorted: [ ");
        for (uint32_t i = 0; i < count; ++i)
            printf("%d ", *(ptr + i));
        printf("]\n");

        std::sort(std::begin(test_data), std::end(test_data));
        for (uint32_t i = 0; i < count; ++i)
        {
            assert(test_data[i] == *(ptr + i));
        }

        vmaUnmapMemory(ctx->allocator, out_buffer.allocation);
    }


    for (int i = 0; i < 2; ++i)
        ctx->destroy_buffer(keyvals_buffers[i]);
    ctx->destroy_buffer(internal_buffer);
}

static void test_indirect(Context* ctx)
{
    uint32_t test_data[] = { 8,   7,   3,   6,   9,   1,   4,   5,   2,  10, };
    uint32_t count = (uint32_t)std::size(test_data);

    RadixSortContext* sort_ctx = radix_sort_context_create(ctx, count);

    BufferDesc desc{};
    desc.size = sizeof(test_data);
    desc.usage_flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    desc.data = test_data;
    Buffer staging = ctx->create_buffer(desc);

    desc.size = sizeof(uint32_t);
    desc.data = &count;
    desc.usage_flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer count_buffer = ctx->create_buffer(desc);

    VkCommandBuffer cmd = ctx->transfer_command_buffer;

    VkHelpers::begin_command_buffer(cmd, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VkBufferCopy buffer_copy{};
    buffer_copy.srcOffset = buffer_copy.dstOffset = 0;
    buffer_copy.size = sizeof(test_data);
    vkCmdCopyBuffer(cmd, staging.buffer, sort_ctx->even_buffer.buffer, 1, &buffer_copy);

    VkHelpers::full_barrier(cmd);

    radix_sort_vk_sort_indirect_info sort_info{};
    sort_info.key_bits = 32;
    sort_info.count = { count_buffer.buffer, 0, VK_WHOLE_SIZE };
    sort_info.keyvals_even = { sort_ctx->even_buffer.buffer, 0, VK_WHOLE_SIZE };
    sort_info.keyvals_odd = { sort_ctx->odd_buffer.buffer, 0, VK_WHOLE_SIZE };
    sort_info.internal = { sort_ctx->internal_buffer.buffer, 0, VK_WHOLE_SIZE };
    sort_info.indirect = { sort_ctx->indirect_buffer.buffer, 0, VK_WHOLE_SIZE };

    VkDescriptorBufferInfo keyvals_sorted{};
    radix_sort_vk_sort_indirect(ctx->radix_sort_instance, &sort_info, ctx->device, ctx->transfer_command_buffer, &keyvals_sorted);

    vkCmdCopyBuffer(cmd, keyvals_sorted.buffer, staging.buffer, 1, &buffer_copy);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmd;

    VK_CHECK(vkQueueSubmit(ctx->graphics_queue, 1, &info, VK_NULL_HANDLE));

    VK_CHECK(vkQueueWaitIdle(ctx->graphics_queue));

    VK_CHECK(vkResetCommandPool(ctx->device, ctx->transfer_command_pool, 0));

    printf("vk-radix-sort indirect test:\n");
    printf("Original: [ ");
    for (uint32_t i = 0; i < count; ++i) printf("%d ", test_data[i]);
    printf(" ]\n");

    std::sort(std::begin(test_data), std::end(test_data));
    printf("Sorted: [ ");
    void* mapped;
    vmaMapMemory(ctx->allocator, staging.allocation, &mapped);
    uint32_t* ptr = (uint32_t*)mapped;
    for (uint32_t i = 0; i < count; ++i)
    {
        assert(test_data[i] == *(ptr + i));
        printf("%d ", *(ptr + i));
    }
    printf(" ]\n");

    vmaUnmapMemory(ctx->allocator, staging.allocation);

    ctx->destroy_buffer(staging);
    ctx->destroy_buffer(count_buffer);

    radix_sort_context_destroy(sort_ctx);
}

void test_radix_sort(Context* ctx)
{
    test_direct(ctx);
    test_indirect(ctx);
}

RadixSortContext* radix_sort_context_create(Context* ctx, uint32_t max_count)
{
    radix_sort_vk_memory_requirements memory_requirements{};
    radix_sort_vk_get_memory_requirements(ctx->radix_sort_instance, max_count, &memory_requirements);

    BufferDesc desc{};
    desc.size = memory_requirements.keyvals_size;
    desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    Buffer keyvals_buffers[2] = {};
    for (int i = 0; i < 2; ++i)
        keyvals_buffers[i] = ctx->create_buffer(desc, memory_requirements.keyvals_alignment);

    desc.allocation_flags = 0;
    desc.size = memory_requirements.internal_size;
    Buffer internal_buffer = ctx->create_buffer(desc, memory_requirements.internal_alignment);

    desc.usage_flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    desc.size = memory_requirements.indirect_size;
    Buffer indirect_buffer = ctx->create_buffer(desc, memory_requirements.indirect_alignment);

    RadixSortContext* sort_ctx = new RadixSortContext();
    sort_ctx->ctx = ctx;
    sort_ctx->even_buffer = keyvals_buffers[0];
    sort_ctx->odd_buffer = keyvals_buffers[1];
    sort_ctx->internal_buffer = internal_buffer;
    sort_ctx->indirect_buffer = indirect_buffer;

    return sort_ctx;
}

void radix_sort_context_destroy(RadixSortContext* ctx)
{
    ctx->ctx->destroy_buffer(ctx->even_buffer);
    ctx->ctx->destroy_buffer(ctx->odd_buffer);
    ctx->ctx->destroy_buffer(ctx->internal_buffer);
    ctx->ctx->destroy_buffer(ctx->indirect_buffer);
    delete ctx;
}

const Buffer& radix_sort_context_get_input(RadixSortContext* ctx)
{
    return ctx->even_buffer;
}

void radix_sort_sort_indirect(RadixSortContext* ctx, uint32_t count, VkCommandBuffer cmd, const Buffer** output_buffer)
{
}

void radix_sort_sort_direct(RadixSortContext* ctx, uint32_t count, VkCommandBuffer cmd, const Buffer** output_buffer)
{
    VkDescriptorBufferInfo keyvals_even{
        ctx->even_buffer.buffer,
        0,
        VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo keyvals_odd{
        ctx->odd_buffer.buffer,
        0,
        VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo internal{
        ctx->internal_buffer.buffer,
        0,
        VK_WHOLE_SIZE
    };
    radix_sort_vk_sort_info_t sort_info{};
    sort_info.key_bits = 32;
    sort_info.count = count;
    sort_info.keyvals_even = keyvals_even;
    sort_info.keyvals_odd = keyvals_odd;
    sort_info.internal = internal;

    VkDescriptorBufferInfo keyvals_sorted{};
    radix_sort_vk_sort(ctx->ctx->radix_sort_instance, &sort_info, ctx->ctx->device, cmd, &keyvals_sorted);
    assert(keyvals_sorted.offset == 0);
    assert(keyvals_sorted.range == VK_WHOLE_SIZE);
    *output_buffer = keyvals_sorted.buffer == ctx->even_buffer.buffer ? &ctx->even_buffer : &ctx->odd_buffer;
}
