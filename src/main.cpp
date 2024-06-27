#include "defines.h"
#include "VkBootstrap.h"
#include "vk_helpers.h"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(embedded_shaders);

constexpr uint32_t window_width = 1280;
constexpr uint32_t window_height = 720;

struct Texture
{
    VkImage image;
    VkImageView view;
    VkImageLayout layout;
    VmaAllocation allocation;
};

struct PipelineBuilder
{
    VkDevice device;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkGraphicsPipelineCreateInfo pipeline_create_info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

    static constexpr uint32_t max_shader_stages = 4;
    VkPipelineShaderStageCreateInfo shader_stage_create_info[max_shader_stages] = {};
    VkShaderModule shader_modules[max_shader_stages] = {};

    VkPipelineVertexInputStateCreateInfo vertex_input_state = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineTessellationStateCreateInfo tesselation_state = { VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };
    VkPipelineViewportStateCreateInfo viewport_state = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rasterization_state = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo multisample_state = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkPipelineColorBlendStateCreateInfo color_blend_state = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    VkPipelineDynamicStateCreateInfo dynamic_state = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    VkPipelineRenderingCreateInfo rendering_create_info = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };

    static constexpr uint32_t max_dynamic_states = 32;
    uint32_t dynamic_state_count = 0;
    VkDynamicState dynamic_states[max_dynamic_states] = {};

    static constexpr uint32_t max_color_blend_attachments = 8;
    VkPipelineColorBlendAttachmentState color_blend_attachments[max_color_blend_attachments] = {};
    VkFormat color_attachment_formats[max_color_blend_attachments] = {};
    uint32_t color_attachment_count = 0;

    PipelineBuilder(VkDevice dev)
        : device(dev)
    {
        pipeline_create_info.pNext = &rendering_create_info;
        pipeline_create_info.pStages = shader_stage_create_info;
        pipeline_create_info.pVertexInputState = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pTessellationState = &tesselation_state;
        pipeline_create_info.pViewportState = &viewport_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pMultisampleState = &multisample_state;
        pipeline_create_info.pDepthStencilState = &depth_stencil_state;
        pipeline_create_info.pColorBlendState = &color_blend_state;
        pipeline_create_info.pDynamicState = &dynamic_state;

        // Set default values

        input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization_state.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization_state.lineWidth = 1.0f;

        multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        color_blend_state.pAttachments = color_blend_attachments;

        dynamic_states[dynamic_state_count++] = VK_DYNAMIC_STATE_VIEWPORT;
        dynamic_states[dynamic_state_count++] = VK_DYNAMIC_STATE_SCISSOR;

        dynamic_state.dynamicStateCount = dynamic_state_count;
        dynamic_state.pDynamicStates = dynamic_states;

        rendering_create_info.pColorAttachmentFormats = color_attachment_formats;
    }

    PipelineBuilder& add_color_attachment(VkFormat format)
    {
        color_blend_attachments[color_attachment_count].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_attachment_formats[color_attachment_count] = format;
        color_attachment_count++;

        color_blend_state.attachmentCount = color_attachment_count;
        rendering_create_info.colorAttachmentCount = color_attachment_count;

        return *this;
    }

    PipelineBuilder& set_layout(VkPipelineLayout layout)
    {
        pipeline_create_info.layout = layout;
        
        return *this;
    }

    PipelineBuilder& set_vertex_shader_spirv(uint32_t* data, size_t size)
    {
        return add_shader_stage_spirv(data, size, VK_SHADER_STAGE_VERTEX_BIT, "vs_main");
    }

    PipelineBuilder& set_fragment_shader_spirv(uint32_t* data, size_t size)
    {
        return add_shader_stage_spirv(data, size, VK_SHADER_STAGE_FRAGMENT_BIT, "fs_main");
    }

    static std::string get_embedded_path(const char* src_path, VkShaderStageFlagBits shader_stage)
    {
        std::string path = "shaders/" + std::string(src_path);
        size_t dot = path.find_last_of('.');
        path = path.substr(0, dot);
        switch (shader_stage)
        {
        case VK_SHADER_STAGE_VERTEX_BIT:
            path.append("_vs_6_6.spv");
            break;
        case VK_SHADER_STAGE_FRAGMENT_BIT:
            path.append("_ps_6_6.spv");
            break;
        case VK_SHADER_STAGE_COMPUTE_BIT:
            path.append("_cs_6_6.spv");
            break;
        default:
            assert(false);
            break;
        }

        return path;
    }

    PipelineBuilder& set_cull_mode(VkCullModeFlagBits cull_mode)
    {
        rasterization_state.cullMode = cull_mode;

        return *this;
    }

    PipelineBuilder& set_vertex_shader_filepath(const char* filepath)
    {
        auto fs = cmrc::embedded_shaders::get_filesystem();
        std::string path = get_embedded_path(filepath, VK_SHADER_STAGE_VERTEX_BIT);
        auto file = fs.open(path);
        assert(file.size() % 4 == 0);
        return set_vertex_shader_spirv((uint32_t*)file.begin(), file.size());
    }

    PipelineBuilder& set_fragment_shader_filepath(const char* filepath)
    {
        auto fs = cmrc::embedded_shaders::get_filesystem();
        std::string path = get_embedded_path(filepath, VK_SHADER_STAGE_FRAGMENT_BIT);
        auto file = fs.open(path);
        assert(file.size() % 4 == 0);
        return set_fragment_shader_spirv((uint32_t*)file.begin(), file.size());
    }

    PipelineBuilder& add_shader_stage_spirv(uint32_t* data, size_t size, VkShaderStageFlagBits shader_stage, const char* entry_point)
    {
        uint32_t stage_count = pipeline_create_info.stageCount;
        assert(stage_count < max_shader_stages);

        VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        info.codeSize = size;
        info.pCode = data;
        VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &shader_modules[stage_count]));

        VkPipelineShaderStageCreateInfo& stage_info = shader_stage_create_info[stage_count];

        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = shader_stage;
        stage_info.module = shader_modules[stage_count];
        stage_info.pName = entry_point;

        pipeline_create_info.stageCount++;

        return *this;
    }

    VkPipeline build()
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_create_info, nullptr, &pipeline));
        for (uint32_t i = 0; i < pipeline_create_info.stageCount; ++i)
        {
            vkDestroyShaderModule(device, shader_modules[i], nullptr);
        }
        return pipeline;
    }
};

struct Context
{
    static constexpr uint32_t frames_in_flight = 2;
    SDL_Window* window;
    vkb::Instance instance;
    VkSurfaceKHR surface;
    vkb::PhysicalDevice physical_device;
    vkb::Device device;
    uint32_t graphics_queue_family_index;
    VkQueue graphics_queue;
    uint32_t transfer_queue_family_index;
    VkQueue transfer_queue;
    vkb::Swapchain swapchain;

    VmaAllocator allocator;

    std::vector<Texture> swapchain_textures;
    uint32_t swapchain_image_index = 0;

    VkCommandPool transfer_command_pool;

    uint32_t frame_index = 0;
    VkCommandPool command_pools[frames_in_flight];
    VkFence frame_fences[frames_in_flight];
    VkSemaphore image_acquired_semaphore[frames_in_flight];
    VkSemaphore rendering_finished_semaphore[frames_in_flight];

    void init()
    {
        SDL_Init(SDL_INIT_VIDEO);

        window = SDL_CreateWindow("Gigasticle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_VULKAN);

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

        VkPhysicalDeviceVulkan13Features vulkan_13_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        vulkan_13_features.dynamicRendering = VK_TRUE;
        vulkan_13_features.synchronization2 = VK_TRUE;

        vkb::PhysicalDeviceSelector phys_device_selector(instance);
        auto physical_device_selector_return = phys_device_selector
            .set_surface(surface)
            .set_required_features_13(vulkan_13_features)
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
        graphics_queue_family_index = device.get_queue_index(vkb::QueueType::graphics).value();
        graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
        transfer_queue_family_index = device.get_queue_index(vkb::QueueType::transfer).value();
        transfer_queue = device.get_queue(vkb::QueueType::transfer).value();

        vkb::SwapchainBuilder swapchain_builder{ device };
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

            VkFenceCreateInfo fence_info{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &frame_fences[i]));

            VkSemaphoreCreateInfo semaphore_info{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &image_acquired_semaphore[i]));
            VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &rendering_finished_semaphore[i]));
        }

        transfer_command_pool = VkHelpers::create_command_pool(device, transfer_queue_family_index);

        { // Create VMA allocator
            VmaVulkanFunctions funcs{};
            funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
            funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
            VmaAllocatorCreateInfo allocator_info{};
            allocator_info.device = device;
            allocator_info.physicalDevice = physical_device;
            allocator_info.instance = instance;
            allocator_info.pVulkanFunctions = &funcs;
            VK_CHECK(vmaCreateAllocator(&allocator_info, &allocator));
        }
    }

    void shutdown()
    {
        VK_CHECK(vkDeviceWaitIdle(device));

        vmaDestroyAllocator(allocator);

        for (auto& t : swapchain_textures)
        {
            vkDestroyImageView(device, t.view, nullptr);
        }
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

    VkCommandBuffer begin_frame()
    {
        VK_CHECK(vkWaitForFences(device, 1, &frame_fences[frame_index], VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &frame_fences[frame_index]));

        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_acquired_semaphore[frame_index], VK_NULL_HANDLE, &swapchain_image_index);

        VK_CHECK(vkResetCommandPool(device, command_pools[frame_index], 0));

        VkCommandBufferAllocateInfo info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        info.commandPool = command_pools[frame_index];
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(device, &info, &cmd));

        VkCommandBufferBeginInfo cmd_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cmd_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_info));

        VkImageMemoryBarrier2 image_barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        image_barrier.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        image_barrier.srcAccessMask = 0;
        image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        image_barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    void end_frame(VkCommandBuffer command_buffer)
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
        image_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    Texture& get_swapchain_texture()
    {
        return swapchain_textures[swapchain_image_index];
    }

    Texture create_texture_hdr(float* data, int width, int height)
    {
        VkImageCreateInfo image_create_info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

        /*
        *     VkStructureType          sType;
    const void*              pNext;
    VkImageCreateFlags       flags;
    VkImageType              imageType;
    VkFormat                 format;
    VkExtent3D               extent;
    uint32_t                 mipLevels;
    uint32_t                 arrayLayers;
    VkSampleCountFlagBits    samples;
    VkImageTiling            tiling;
    VkImageUsageFlags        usage;
    VkSharingMode            sharingMode;
    uint32_t                 queueFamilyIndexCount;
    const uint32_t*          pQueueFamilyIndices;
    VkImageLayout            initialLayout;
        */
        VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

        image_create_info.imageType = VK_IMAGE_TYPE_2D;
        image_create_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        image_create_info.extent = { (uint32_t)width, (uint32_t)height, 1u };
        image_create_info.mipLevels = 1;
        image_create_info.arrayLayers = 1;
        image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo allocation_create_info{};
        allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VK_CHECK(vmaCreateImage(allocator, &image_create_info, &allocation_create_info, &image, &allocation, nullptr));
        
        VkImageViewCreateInfo image_view_info{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        image_view_info.image = image;
        image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        image_view_info.format = format;
        image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_view_info.subresourceRange.baseArrayLayer = 0;
        image_view_info.subresourceRange.baseMipLevel = 0;
        image_view_info.subresourceRange.layerCount = 1;
        image_view_info.subresourceRange.levelCount = 1;

        VK_CHECK(vkCreateImageView(device, &image_view_info, nullptr, &view));

        VkBufferCreateInfo buffer_info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
       /*    VkStructureType        sType;
    const void*            pNext;
    VkBufferCreateFlags    flags;
    VkDeviceSize           size;
    VkBufferUsageFlags     usage;
    VkSharingMode          sharingMode;
    uint32_t               queueFamilyIndexCount;
    const uint32_t*        pQueueFamilyIndices;*/
        const uint32_t n_channels = 4;
        uint32_t required_size = sizeof(float) * width * height * n_channels;
        buffer_info.size = required_size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        allocation_create_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation buffer_allocation;
        VK_CHECK(vmaCreateBuffer(allocator, &buffer_info, &allocation_create_info, &staging_buffer, &buffer_allocation, nullptr));

        void* mapped;
        vmaMapMemory(allocator, buffer_allocation, &mapped);
        memcpy(mapped, data, required_size);
        vmaUnmapMemory(allocator, buffer_allocation);

        VkCommandBufferAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        alloc_info.commandPool = transfer_command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(device, &alloc_info, &cmd));

        VkHelpers::begin_command_buffer(cmd, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        {
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                image
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep_info);
        }

        VkBufferImageCopy2 region{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { (uint32_t)width, (uint32_t)height, 1 };
        region.imageOffset = { 0, 0, 0 };

        VkCopyBufferToImageInfo2 copy_image{ VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2 };
        copy_image.srcBuffer = staging_buffer;
        copy_image.dstImage = image;
        copy_image.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copy_image.regionCount = 1;
        copy_image.pRegions = &region;

        vkCmdCopyBufferToImage2(cmd, &copy_image);

        {
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                image
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep_info);
        }

        vkEndCommandBuffer(cmd);

        VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        info.commandBufferCount = 1;
        info.pCommandBuffers = &cmd;

        VK_CHECK(vkQueueSubmit(transfer_queue, 1, &info, VK_NULL_HANDLE));

        VK_CHECK(vkQueueWaitIdle(transfer_queue));

        VK_CHECK(vkResetCommandPool(device, transfer_command_pool, 0));
        vmaDestroyBuffer(allocator, staging_buffer, buffer_allocation);

        Texture tex{};
        tex.image = image;
        tex.view = view;
        tex.allocation = allocation;

        return tex;
    }
};

Context ctx;

int main()
{
    ctx.init();

    bool running = true;

    PipelineBuilder pipeline_builder(ctx.device);

    VkPipelineLayout pipeline_layout;
    {
        VkPipelineLayoutCreateInfo layout_create_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        /*    VkStructureType                 sType;
        const void*                     pNext;
        VkPipelineLayoutCreateFlags     flags;
        uint32_t                        setLayoutCount;
        const VkDescriptorSetLayout*    pSetLayouts;
        uint32_t                        pushConstantRangeCount;
        const VkPushConstantRange*      pPushConstantRanges;*/
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &layout_create_info, nullptr, &pipeline_layout));
    }

    VkPipeline pipeline = pipeline_builder
        .set_vertex_shader_filepath("triangle.hlsl")
        .set_fragment_shader_filepath("triangle.hlsl")
        .add_color_attachment(ctx.swapchain.image_format)
        .set_layout(pipeline_layout)
        .set_cull_mode(VK_CULL_MODE_NONE)
        .build();

    int x, y, c;
    float* data = stbi_loadf("data/grace_probe.hdr", &x, &y, &c, 4);
    Texture hdr = ctx.create_texture_hdr(data, x, y);

    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            switch (e.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
            {
                if (e.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
                {
                    running = false;
                }
                break;
            }
            default:
                break;
            }
        }

        VkCommandBuffer command_buffer = ctx.begin_frame();
        Texture& swapchain_texture = ctx.get_swapchain_texture();

        VkRenderingAttachmentInfo color_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        color_info.imageView = swapchain_texture.view;
        color_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_info.clearValue.color = { 1.0f, 0.0f, 1.0f, 1.0f };

        VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        rendering_info.renderArea = { {0, 0}, {window_width, window_height} };
        rendering_info.layerCount = 1;
        rendering_info.viewMask = 0;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_info;

        vkCmdBeginRendering(command_buffer, &rendering_info);

        VkRect2D scissor = { {0, 0}, {window_width, window_height} };
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        VkViewport viewport = {0.0f, (float)window_height, (float)window_width, -(float)window_height, 0.0f, 1.0f};
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);

        vkCmdEndRendering(command_buffer);

        ctx.end_frame(command_buffer);
    }

    vkDeviceWaitIdle(ctx.device);
    vkDestroyImageView(ctx.device, hdr.view, nullptr);
    vmaDestroyImage(ctx.allocator, hdr.image, hdr.allocation);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);

    ctx.shutdown();

    return 0;
}