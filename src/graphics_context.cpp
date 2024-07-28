#define VOLK_IMPLEMENTATION
#include "graphics_context.h"
#include "vk_helpers.h"
#include "buffer.h"
#include "misc.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl2.h"
#include "imgui/imgui_impl_vulkan.h"

constexpr uint32_t MAX_BINDLESS_RESOURCES = 1024;

void Context::init(int window_width, int window_height)
{
    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow("Gigasticle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_VULKAN);

    VK_CHECK(volkInitialize());

    vkb::InstanceBuilder instance_builder;
    instance_builder.require_api_version(1, 3, 0);
    instance_builder.set_app_name("Gigasticle");
    instance_builder.request_validation_layers();
    instance_builder.enable_extensions({
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME
        });
    instance_builder.set_debug_callback(
        [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData)
        -> VkBool32 {
            auto severity = vkb::to_string_message_severity(messageSeverity);
            auto type = vkb::to_string_message_type(messageType);
            LOG_ERROR("[%s: %s] %s\n", severity, type, pCallbackData->pMessage);
            if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
            {
                assert(false);
            }
            return VK_FALSE;
        }
    );

    vkb::Result<vkb::Instance> instance_result = instance_builder.build();
    if (!instance_result)
    {
        LOG_ERROR("Failed to create vulkan instance. Error: %s", instance_result.error().message());
        exit(-1);
    }

    instance = instance_result.value();
    volkLoadInstanceOnly(instance.instance);

    auto system_info_ret = vkb::SystemInfo::get_system_info();
    if (!system_info_ret) {
        LOG_ERROR("%s\n", system_info_ret.error().message());
        return exit(-1);
    }
    auto system_info = system_info_ret.value();

    if (!SDL_Vulkan_CreateSurface(window, instance, &surface))
    {
        LOG_ERROR("Failed to create Vulkan surface");
        exit(-1);
    }

    VkPhysicalDeviceVulkan13Features vulkan_13_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    vulkan_13_features.dynamicRendering = VK_TRUE;
    vulkan_13_features.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan_12_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    vulkan_12_features.bufferDeviceAddress = VK_TRUE;
    vulkan_12_features.descriptorIndexing = VK_TRUE;
    vulkan_12_features.descriptorBindingPartiallyBound = VK_TRUE;
    vulkan_12_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    vulkan_12_features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    vulkan_12_features.runtimeDescriptorArray = VK_TRUE;

    VkPhysicalDeviceVulkan11Features vulkan_11_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    vulkan_11_features.multiview = VK_TRUE;

    VkPhysicalDeviceFeatures features{};
    features.shaderInt64 = VK_TRUE;
    features.samplerAnisotropy = VK_TRUE;

    vkb::PhysicalDeviceSelector phys_device_selector(instance);
    auto physical_device_selector_return = phys_device_selector
        .set_surface(surface)
        .set_required_features(features)
        .set_required_features_11(vulkan_11_features)
        .set_required_features_12(vulkan_12_features)
        .set_required_features_13(vulkan_13_features)
        .add_required_extensions({ VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME })
        .select();
    if (!physical_device_selector_return) {
        // Handle error
        LOG_ERROR("Failed to create Vulkan physical device!");
        exit(-1);
    }

    physical_device = physical_device_selector_return.value();
    LOG_INFO("Selected physical device: %s", physical_device.name.c_str());
    std::vector<std::string> available_device_exts = physical_device.get_available_extensions();
    LOG_INFO("Available device extensions:");
    for (const auto& e : available_device_exts)
    {
        LOG_INFO("%s", e.c_str());
    }

    vkb::DeviceBuilder device_builder{ physical_device };
    auto dev_ret = device_builder.build();
    if (!dev_ret) {
        // error
        LOG_ERROR("Failed to create Vulkan device!");
        exit(-1);
    }
    device = dev_ret.value();
    volkLoadDevice(device.device);

    graphics_queue_family_index = device.get_queue_index(vkb::QueueType::graphics).value();
    graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
    transfer_queue_family_index = device.get_queue_index(vkb::QueueType::graphics).value();
    transfer_queue = device.get_queue(vkb::QueueType::graphics).value();

    vkb::SwapchainBuilder swapchain_builder{ device };
    swapchain_builder.set_desired_format({ VK_FORMAT_B8G8R8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR });
    swapchain_builder.set_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto swap_ret = swapchain_builder.build();
    if (!swap_ret) {
        LOG_ERROR("Failed to create swapchain!");
        exit(-1);
    }
    swapchain = swap_ret.value();
    auto swapchain_images = swapchain.get_images().value();
    auto swapchain_image_views = swapchain.get_image_views().value();
    assert(swapchain_images.size() == swapchain_image_views.size());

    swapchain_textures.resize(swapchain_images.size());
    for (size_t i = 0; i < swapchain_images.size(); ++i)
    {
        Texture& tex = swapchain_textures[i];
        tex.image = swapchain_images[i];
        tex.view = swapchain_image_views[i];
        tex.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        assert(tex.image);
        assert(tex.view);
    }

    LOG_DEBUG("Swapchain format: %s", string_VkFormat(swapchain.image_format));

    for (uint32_t i = 0; i < frames_in_flight; ++i)
    {
        VkCommandPoolCreateInfo info{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        info.queueFamilyIndex = graphics_queue_family_index;
        VK_CHECK(vkCreateCommandPool(device, &info, nullptr, &command_pools[i]));

        VkCommandBufferAllocateInfo cmd_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmd_info.commandPool = command_pools[i];
        cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_info.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmd_info, &command_buffers[i]));

        VkFenceCreateInfo fence_info{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &frame_fences[i]));

        VkSemaphoreCreateInfo semaphore_info{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &image_acquired_semaphore[i]));
        VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &rendering_finished_semaphore[i]));
    }

    {
        transfer_command_pool = VkHelpers::create_command_pool(device, graphics_queue_family_index);

        VkCommandBufferAllocateInfo cmd_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmd_info.commandPool = transfer_command_pool;
        cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_info.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmd_info, &transfer_command_buffer));
    }

    { // Create VMA allocator
        VmaVulkanFunctions funcs{};
        funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        VmaAllocatorCreateInfo allocator_info{};
        allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        allocator_info.device = device;
        allocator_info.physicalDevice = physical_device;
        allocator_info.instance = instance;
        allocator_info.pVulkanFunctions = &funcs;
        VK_CHECK(vmaCreateAllocator(&allocator_info, &allocator));
    }

    { // Bindless descriptor pool
        VkDescriptorPoolSize pool_sizes_bindless[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_BINDLESS_RESOURCES},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_BINDLESS_RESOURCES}
        };

        VkDescriptorPoolCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        info.maxSets = MAX_BINDLESS_RESOURCES * std::size(pool_sizes_bindless);
        info.poolSizeCount = (uint32_t)std::size(pool_sizes_bindless);
        info.pPoolSizes = pool_sizes_bindless;
        VK_CHECK(vkCreateDescriptorPool(device, &info, nullptr, &bindless_descriptor_pool));
    }

    { // Imgui descriptor pool
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
        };

        VkDescriptorPoolCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        info.maxSets = MAX_BINDLESS_RESOURCES * std::size(pool_sizes);
        info.poolSizeCount = (uint32_t)std::size(pool_sizes);
        info.pPoolSizes = pool_sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &info, nullptr, &imgui_descriptor_pool));
    }

    { // Bindless descriptor set layout
        VkDescriptorSetLayoutBinding bindings[2];
        VkDescriptorSetLayoutBinding& image_sampler_binding = bindings[0];
        image_sampler_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        image_sampler_binding.descriptorCount = MAX_BINDLESS_RESOURCES;
        image_sampler_binding.binding = 0;
        image_sampler_binding.stageFlags = VK_SHADER_STAGE_ALL;
        VkDescriptorSetLayoutBinding& storage_image_binding = bindings[1];
        storage_image_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        storage_image_binding.descriptorCount = MAX_BINDLESS_RESOURCES;
        storage_image_binding.binding = 1;
        storage_image_binding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layout_info.bindingCount = std::size(bindings);
        layout_info.pBindings = bindings;
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;
        VkDescriptorBindingFlags bindless_flags =
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;
        VkDescriptorBindingFlags binding_flags[2];
        binding_flags[0] = bindless_flags;
        binding_flags[1] = bindless_flags;
        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extended_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT };
        extended_info.bindingCount = std::size(bindings);
        extended_info.pBindingFlags = binding_flags;
        layout_info.pNext = &extended_info;

        VK_CHECK(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &bindless_descriptor_set_layout));
    }

    {
        VkDescriptorSetAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        alloc_info.descriptorPool = bindless_descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &bindless_descriptor_set_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &bindless_descriptor_set));
    }
}

void Context::shutdown()
{
    VK_CHECK(vkDeviceWaitIdle(device));

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    vmaDestroyAllocator(allocator);

    for (auto& t : swapchain_textures)
    {
        vkDestroyImageView(device, t.view, nullptr);
    }
    vkDestroyDescriptorPool(device, bindless_descriptor_pool, nullptr);
    vkDestroyDescriptorPool(device, imgui_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, bindless_descriptor_set_layout, nullptr);
    for (uint32_t i = 0; i < frames_in_flight; ++i)
    {
        vkDestroyCommandPool(device, command_pools[i], nullptr);
        vkDestroyFence(device, frame_fences[i], nullptr);
        vkDestroySemaphore(device, image_acquired_semaphore[i], nullptr);
        vkDestroySemaphore(device, rendering_finished_semaphore[i], nullptr);
    }
    vkDestroyCommandPool(device, transfer_command_pool, nullptr);
    vkb::destroy_swapchain(swapchain);
    vkb::destroy_device(device);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkb::destroy_instance(instance);

    SDL_DestroyWindow(window);
    SDL_Quit();
}

VkCommandBuffer Context::begin_frame()
{
    VK_CHECK(vkWaitForFences(device, 1, &frame_fences[frame_index], VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device, 1, &frame_fences[frame_index]));

    vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_acquired_semaphore[frame_index], VK_NULL_HANDLE, &swapchain_image_index);

    VK_CHECK(vkResetCommandPool(device, command_pools[frame_index], 0));

    VkCommandBuffer cmd = command_buffers[frame_index];

    VkCommandBufferBeginInfo cmd_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cmd_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_info));

    VkImageMemoryBarrier2 image_barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    image_barrier.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    image_barrier.srcAccessMask = 0;
    image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.image = swapchain_textures[swapchain_image_index].image;
    image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_barrier.subresourceRange.baseArrayLayer = 0;
    image_barrier.subresourceRange.baseMipLevel = 0;
    image_barrier.subresourceRange.layerCount = 1;
    image_barrier.subresourceRange.levelCount = 1;

    VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &image_barrier;
    vkCmdPipelineBarrier2(cmd, &dep_info);

    return cmd;
}

void Context::end_frame(VkCommandBuffer command_buffer)
{
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    info.commandBufferCount = 1;
    info.pCommandBuffers = &command_buffer;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &image_acquired_semaphore[frame_index];
    info.pWaitDstStageMask = &wait_stage;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &rendering_finished_semaphore[frame_index];

    VkImageMemoryBarrier2 image_barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    image_barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    image_barrier.image = swapchain_textures[swapchain_image_index].image;
    image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_barrier.subresourceRange.baseArrayLayer = 0;
    image_barrier.subresourceRange.baseMipLevel = 0;
    image_barrier.subresourceRange.layerCount = 1;
    image_barrier.subresourceRange.levelCount = 1;

    VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &image_barrier;
    vkCmdPipelineBarrier2(command_buffer, &dep_info);

    vkEndCommandBuffer(command_buffer);

    VK_CHECK(vkQueueSubmit(graphics_queue, 1, &info, frame_fences[frame_index]));

    VkPresentInfoKHR present_info{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &rendering_finished_semaphore[frame_index];
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain.swapchain;
    present_info.pImageIndices = &swapchain_image_index;
    VK_CHECK(vkQueuePresentKHR(graphics_queue, &present_info));

    frame_index = (frame_index + 1) % frames_in_flight;
}


bool Context::create_texture(Texture& texture, uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageType image_type, VkImageUsageFlags usage, uint32_t mip_levels, uint32_t array_layers)
{
    VkImageCreateInfo image_create_info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

    image_create_info.imageType = image_type;
    image_create_info.format = format;
    image_create_info.extent = { (uint32_t)width, (uint32_t)height, (uint32_t)depth };
    image_create_info.mipLevels = mip_levels;
    image_create_info.arrayLayers = array_layers;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = usage;

    VmaAllocationCreateInfo allocation_create_info{};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vmaCreateImage(allocator, &image_create_info, &allocation_create_info, &image, &allocation, nullptr));

    VkImageViewCreateInfo image_view_info{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    image_view_info.image = image;
    image_view_info.viewType = array_layers == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    image_view_info.format = format;
    image_view_info.subresourceRange.aspectMask = determine_image_aspect(format);
    image_view_info.subresourceRange.baseArrayLayer = 0;
    image_view_info.subresourceRange.baseMipLevel = 0;
    image_view_info.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    image_view_info.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

    VK_CHECK(vkCreateImageView(device, &image_view_info, nullptr, &view));

    texture.allocation = allocation;
    texture.image = image;
    texture.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    texture.view = view;

    return true;
}

bool Context::create_textures(Texture* textures, uint32_t count)
{
    VkHelpers::begin_command_buffer(transfer_command_buffer, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    std::vector<Buffer> staging_buffers(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        Texture& t = textures[i];
        uint32_t mip_count = get_mip_count(t.width, t.height);
        VkImageUsageFlags image_usage_flags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (!create_texture(t, t.width, t.height, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, image_usage_flags, mip_count, 1))
        {
            return false;
        }

        const uint32_t n_channels = 4;
        uint32_t required_size = t.width * t.height * n_channels;

        BufferDesc desc{};
        desc.size = required_size;
        desc.usage_flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        desc.data = t.source;
        Buffer& staging_buffer = staging_buffers[i];
        staging_buffer = create_buffer(desc);

        {
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                t.image
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(transfer_command_buffer, &dep_info);
        }

        VkBufferImageCopy2 region{ VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2 };
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { (uint32_t)t.width, (uint32_t)t.height, 1 };
        region.imageOffset = { 0, 0, 0 };

        VkCopyBufferToImageInfo2 copy_image{ VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2 };
        copy_image.srcBuffer = staging_buffer.buffer;
        copy_image.dstImage = t.image;
        copy_image.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copy_image.regionCount = 1;
        copy_image.pRegions = &region;

        vkCmdCopyBufferToImage2(transfer_command_buffer, &copy_image);

        {
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                t.image
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(transfer_command_buffer, &dep_info);
        }

        t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        int32_t width = t.width;
        int32_t height = t.height;
        for (uint32_t i = 1; i < mip_count; ++i)
        {
            int32_t next_width = std::max(width >> 1, 1);
            int32_t next_height = std::max(height >> 1, 1);

            { // Transition to transfer dst optimal
                VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    0,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    0,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    t.image,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    i, 1
                );

                VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dep_info.imageMemoryBarrierCount = 1;
                dep_info.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(transfer_command_buffer, &dep_info);
            }
            VkImageBlit region{};
            region.srcOffsets[0] = { 0, 0, 0 };
            region.srcOffsets[1] = { width, height, 1 };
            region.dstOffsets[0] = { 0, 0, 0 };
            region.dstOffsets[1] = { next_width, next_height, 1 };
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.mipLevel = i - 1;
            region.srcSubresource.baseArrayLayer = 0;
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.dstSubresource.mipLevel = i;
            region.dstSubresource.baseArrayLayer = 0;
            region.dstSubresource.layerCount = 1;
            vkCmdBlitImage(transfer_command_buffer, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

            { // Transition to transfer src optimal
                VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    0,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    0,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    t.image,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    i, 1
                );

                VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dep_info.imageMemoryBarrierCount = 1;
                dep_info.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(transfer_command_buffer, &dep_info);
            }

            width = next_width;
            height = next_height;
        }

        { // Transition to transfer src optimal
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                t.image,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, VK_REMAINING_MIP_LEVELS
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(transfer_command_buffer, &dep_info);
        }
    }

    vkEndCommandBuffer(transfer_command_buffer);

    VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    info.commandBufferCount = 1;
    info.pCommandBuffers = &transfer_command_buffer;

    VK_CHECK(vkQueueSubmit(transfer_queue, 1, &info, VK_NULL_HANDLE));

    VK_CHECK(vkQueueWaitIdle(transfer_queue));

    VK_CHECK(vkResetCommandPool(device, transfer_command_pool, 0));

    for (auto& buffer : staging_buffers)
    {
        vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
    }

    return true;
}

Buffer Context::create_buffer(const BufferDesc& desc)
{
    VkBufferCreateInfo buffer_info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    buffer_info.size = desc.size;
    buffer_info.usage = desc.usage_flags;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    allocation_info.flags = desc.allocation_flags;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation;
    VK_CHECK(vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer, &allocation, nullptr));

    if (desc.data)
    {
        assert(desc.allocation_flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        void* mapped;
        vmaMapMemory(allocator, allocation, &mapped);
        memcpy(mapped, desc.data, desc.size);
        vmaUnmapMemory(allocator, allocation);
    }

    Buffer result{};
    result.buffer = buffer;
    result.allocation = allocation;
    return result;
}

VkDeviceAddress Context::buffer_device_address(const Buffer& buffer)
{
    VkBufferDeviceAddressInfo info{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    info.buffer = buffer.buffer;
    return vkGetBufferDeviceAddress(device, &info);
}