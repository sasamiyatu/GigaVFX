#include "defines.h"
#include "VkBootstrap.h"
#include "vk_helpers.h"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define VOLK_IMPLEMENTATION
#include "Volk/volk.h"

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(embedded_shaders);

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/euler_angles.hpp"

//#define SPIRV_REFLECT_USE_SYSTEM_SPIRV_H
#include "spirv_reflect.h"
#include "spirv_reflect.c"

constexpr uint32_t window_width = 1280;
constexpr uint32_t window_height = 720;

constexpr uint32_t MAX_BINDLESS_RESOURCES = 1024;

#define CGLTF_FLOAT_COUNT(accessor) (cgltf_num_components(accessor->type) * accessor->count)
#define VECTOR_SIZE_BYTES(x) (x.size() * sizeof(x[0]))


struct CameraState
{
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    float fov = glm::radians(75.0f);
    float znear = 0.1f;
    float zfar = 100.0f;
};

struct Texture
{
    uint8_t* source;
    int width;
    int height;

    VkFormat format;
    VkImage image;
    VkImageView view;
    VkImageLayout layout;
    VmaAllocation allocation;
};

struct BufferDesc
{
    size_t size;
    VkBufferUsageFlags usage_flags;
    VmaAllocationCreateFlags allocation_flags;
    void* data;
};

struct Buffer
{
    VkBuffer buffer;
    VmaAllocation allocation;

    operator bool() const { return buffer != VK_NULL_HANDLE; }
};


struct Pipeline
{
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layouts[4] = {};
    uint32_t descriptor_set_count = 0;
};

struct PipelineBuilder
{
    VkDevice device;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkGraphicsPipelineCreateInfo pipeline_create_info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

    static constexpr uint32_t max_shader_stages = 4;
    VkPipelineShaderStageCreateInfo shader_stage_create_info[max_shader_stages] = {};
    VkShaderModule shader_modules[max_shader_stages] = {};

    struct
    {
        uint32_t* spirv;
        uint32_t size;
    } shader_sources[max_shader_stages];

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

    VkDescriptorSetLayout set_layouts[4] = {};

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

    PipelineBuilder& set_depth_format(VkFormat format)
    {
        rendering_create_info.depthAttachmentFormat = format;

        return *this;
    }

    PipelineBuilder& set_depth_test(VkBool32 enabled)
    {
        depth_stencil_state.depthTestEnable = enabled;

        return *this;
    }

    PipelineBuilder& set_depth_write(VkBool32 enabled)
    {
        depth_stencil_state.depthWriteEnable = enabled;

        return *this;
    }

    PipelineBuilder& set_depth_compare_op(VkCompareOp op)
    {
        depth_stencil_state.depthCompareOp = op;

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

        shader_sources[stage_count].spirv = data;
        shader_sources[stage_count].size = size;

        pipeline_create_info.stageCount++;

        return *this;
    }

    PipelineBuilder& set_descriptor_set_layout(uint32_t set_index, VkDescriptorSetLayout layout)
    {
        set_layouts[set_index] = layout;
        return *this;
    }

    Pipeline build()
    {
        Pipeline pp{};

        if (pipeline_create_info.layout == VK_NULL_HANDLE)
        { // Create the layout
            std::vector<VkDescriptorSetLayoutBinding> bindings[4];
            uint32_t descriptor_set_count = 0;
            uint32_t push_constant_size = 0;
            VkShaderStageFlags pc_stage_flags = 0;
            for (uint32_t i = 0; i < pipeline_create_info.stageCount; ++i)
            {
                SpvReflectShaderModule mod;
                SpvReflectResult result = spvReflectCreateShaderModule(shader_sources[i].size, shader_sources[i].spirv, &mod);
                assert(result == SPV_REFLECT_RESULT_SUCCESS);
                descriptor_set_count = std::max(mod.descriptor_set_count, descriptor_set_count);
                uint32_t pc_size = 0;
                for (uint32_t j = 0; j < mod.push_constant_block_count; ++j)
                {
                    pc_size += mod.push_constant_blocks[j].size;
                    pc_stage_flags |= mod.shader_stage;
                }
               
                push_constant_size = std::max(push_constant_size, pc_size);
                for (uint32_t j = 0; j < mod.descriptor_set_count; ++j)
                {
                    if (set_layouts[j] != VK_NULL_HANDLE) continue;
                    SpvReflectDescriptorSet descriptor_set = mod.descriptor_sets[j];
                    bindings[j].resize(std::max(descriptor_set.binding_count, (uint32_t)bindings[j].size()));
                    for (uint32_t k = 0; k < descriptor_set.binding_count; ++k)
                    {
                        SpvReflectDescriptorBinding* binding = descriptor_set.bindings[k];
                        VkDescriptorSetLayoutBinding& out_binding = bindings[j][k];
                        out_binding.binding = binding->binding;
                        out_binding.descriptorCount = binding->count;
                        out_binding.descriptorType = (VkDescriptorType)binding->descriptor_type;
                        out_binding.stageFlags |= mod.shader_stage;
                    }

                }
            }

            for (uint32_t i = 0; i < descriptor_set_count; ++i)
            {
                if (set_layouts[i] != VK_NULL_HANDLE) continue;
                VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
                info.bindingCount = bindings[i].size();
                info.pBindings = bindings[i].data();
                VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set_layouts[i]));
            }

            VkPipelineLayoutCreateInfo info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            info.setLayoutCount = descriptor_set_count;
            info.pSetLayouts = set_layouts;
            VkPushConstantRange range{};
            range.stageFlags = pc_stage_flags;
            range.size = push_constant_size;
            info.pPushConstantRanges = &range;
            info.pushConstantRangeCount = push_constant_size != 0 ? 1 : 0;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            VK_CHECK(vkCreatePipelineLayout(device, &info, nullptr, &layout));
            pipeline_create_info.layout = layout;
            pp.layout = layout;
            pp.descriptor_set_count = descriptor_set_count;
            memcpy(pp.set_layouts, set_layouts, sizeof(VkDescriptorSetLayout) * descriptor_set_count);
        }

        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_create_info, nullptr, &pipeline));
        for (uint32_t i = 0; i < pipeline_create_info.stageCount; ++i)
        {
            vkDestroyShaderModule(device, shader_modules[i], nullptr);
        }
        pp.pipeline = pipeline;

        return pp;
    }
};

VkImageAspectFlagBits determine_image_aspect(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}


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
    VkCommandBuffer transfer_command_buffer;

    VkDescriptorPool bindless_descriptor_pool;
    VkDescriptorSetLayout bindless_descriptor_set_layout;
    VkDescriptorSet bindless_descriptor_set;

    uint32_t frame_index = 0;
    VkCommandPool command_pools[frames_in_flight];
    VkCommandBuffer command_buffers[frames_in_flight];
    VkFence frame_fences[frames_in_flight];
    VkSemaphore image_acquired_semaphore[frames_in_flight];
    VkSemaphore rendering_finished_semaphore[frames_in_flight];

    void init()
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

        VkPhysicalDeviceVulkan13Features vulkan_13_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        vulkan_13_features.dynamicRendering = VK_TRUE;
        vulkan_13_features.synchronization2 = VK_TRUE;

        VkPhysicalDeviceVulkan12Features vulkan_12_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        vulkan_12_features.bufferDeviceAddress = VK_TRUE;
        vulkan_12_features.descriptorIndexing = VK_TRUE;
        vulkan_12_features.descriptorBindingPartiallyBound = VK_TRUE;
        vulkan_12_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        vulkan_12_features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
        vulkan_12_features.runtimeDescriptorArray = VK_TRUE;

        VkPhysicalDeviceFeatures features{};
        features.shaderInt64 = VK_TRUE;

        vkb::PhysicalDeviceSelector phys_device_selector(instance);
        auto physical_device_selector_return = phys_device_selector
            .set_surface(surface)
            .set_required_features(features)
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
            transfer_command_pool = VkHelpers::create_command_pool(device, transfer_queue_family_index);

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

    void shutdown()
    {
        VK_CHECK(vkDeviceWaitIdle(device));

        vmaDestroyAllocator(allocator);

        for (auto& t : swapchain_textures)
        {
            vkDestroyImageView(device, t.view, nullptr);
        }
        vkDestroyDescriptorPool(device, bindless_descriptor_pool, nullptr);
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

    VkCommandBuffer begin_frame()
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

    bool create_texture(Texture& texture, uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageType image_type, VkImageUsageFlags usage, uint32_t mip_levels = 1, uint32_t array_layers = 1)
    {
        VkImageCreateInfo image_create_info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

        image_create_info.imageType = image_type;
        image_create_info.format = format;
        image_create_info.extent = { (uint32_t)width, (uint32_t)height, (uint32_t)depth};
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
        image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
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

    bool create_textures(Texture* textures, uint32_t count)
    {
        VkHelpers::begin_command_buffer(transfer_command_buffer, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        std::vector<Buffer> staging_buffers(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            Texture& t = textures[i];
            if (!create_texture(t, t.width, t.height, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, 1))
            {
                exit(1);
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
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    t.image
                );

                VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dep_info.imageMemoryBarrierCount = 1;
                dep_info.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(transfer_command_buffer, &dep_info);
            }

            t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
    }

    Buffer create_buffer(const BufferDesc& desc)
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

    VkDeviceAddress buffer_device_address(const Buffer& buffer)
    {
        VkBufferDeviceAddressInfo info{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        info.buffer = buffer.buffer;
        return vkGetBufferDeviceAddress(device, &info);
    }
};

Context ctx;

struct Mesh
{
    struct Primitive
    {
        uint32_t material;
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

struct Material
{
    glm::vec4 basecolor_factor;
    float roughness_factor;
    float metallic_factor;
    
    int basecolor_texture;
    int metallic_roughness_texture;
    int normal_texture;
};

struct ShaderGlobals
{
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 viewprojection;
};

std::vector<Mesh> meshes;

template <typename F>
void traverse_tree(const cgltf_node* node, F&& f)
{
    f(node);
    for (size_t i = 0; i < node->children_count; ++i)
    {
        traverse_tree(node->children[i], std::forward<F>(f));
    }
}

int main()
{
    ctx.init();

    bool running = true;

    PipelineBuilder pipeline_builder(ctx.device);

    VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout _set_layout = VK_NULL_HANDLE;
#if 0
    {

        VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
            {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        desc_info.bindingCount = (uint32_t)std::size(bindings);
        desc_info.pBindings = bindings;
        desc_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        /*typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType                        sType;
    const void*                            pNext;
    VkDescriptorSetLayoutCreateFlags       flags;
    uint32_t                               bindingCount;
    const VkDescriptorSetLayoutBinding*    pBindings;
} VkDescriptorSetLayoutCreateInfo;*/

        VkPipelineLayoutCreateInfo layout_create_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layout_create_info.setLayoutCount = 1;
        layout_create_info.pSetLayouts = &set_layout;
        VkPushConstantRange range{};
        range.size = 128;
        range.offset = 0;
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        layout_create_info.pushConstantRangeCount = 1;
        layout_create_info.pPushConstantRanges = &range;
        /*    VkStructureType                 sType;
        const void*                     pNext;
        VkPipelineLayoutCreateFlags     flags;
        uint32_t                        setLayoutCount;
        const VkDescriptorSetLayout*    pSetLayouts;
        uint32_t                        pushConstantRangeCount;
        const VkPushConstantRange*      pPushConstantRanges;*/
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &desc_info, nullptr, &set_layout));
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &layout_create_info, nullptr, &pipeline_layout));
    }
#endif

    VkSampler sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VK_CHECK(vkCreateSampler(ctx.device, &info, nullptr, &sampler));
    }

    Texture depth_texture;
    ctx.create_texture(depth_texture, window_width, window_height, 1u, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

    Pipeline pipeline = pipeline_builder
        .set_vertex_shader_filepath("forward.hlsl")
        .set_fragment_shader_filepath("forward.hlsl")
        .add_color_attachment(ctx.swapchain.image_format)
        //.set_layout(pipeline_layout)
        .set_cull_mode(VK_CULL_MODE_NONE)
        .set_depth_format(VK_FORMAT_D32_SFLOAT)
        .set_depth_test(VK_TRUE)
        .set_depth_write(VK_TRUE)
        .set_depth_compare_op(VK_COMPARE_OP_LESS)
        .set_descriptor_set_layout(1, ctx.bindless_descriptor_set_layout)
        .build();

    int x, y, c;
    //float* data = stbi_loadf("data/grace_probe.hdr", &x, &y, &c, 4);
    //Texture hdr = ctx.create_texture_hdr(data, x, y);

    const char* gltf_path = "D:/Projects/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf";
    //const char* gltf_path = "D:/Projects/glTF-Sample-Models/2.0/Box/glTF/Box.gltf";
    cgltf_options opt{};
    cgltf_data* gltf_data = nullptr;
    cgltf_result res = cgltf_parse_file(&opt, gltf_path, &gltf_data);
    if (res != cgltf_result_success)
    {
        LOG_ERROR("Failed to load glTF!");
        exit(-1);
    }

    res = cgltf_load_buffers(&opt, gltf_data, gltf_path);
    if (res != cgltf_result_success)
    {
        LOG_ERROR("Failed to load buffers from glTF!");
        exit(-1);
    }

    meshes.resize(gltf_data->meshes_count);
    for (size_t i = 0; i < gltf_data->meshes_count; ++i)
    {
        const cgltf_mesh& m = gltf_data->meshes[i];
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> normal;
        std::vector<glm::vec4> tangent;
        std::vector<glm::vec2> texcoord0;
        std::vector<glm::vec2> texcoord1;
        std::vector<uint32_t> indices;

        Mesh& mesh = meshes[i];
        mesh.primitives.resize(m.primitives_count);

        for (size_t j = 0; j < m.primitives_count; ++j)
        {
            const cgltf_primitive& p = m.primitives[j];
            assert(p.indices);

            Mesh::Primitive& primitive = mesh.primitives[j];

            uint32_t first_vertex = position.size();
            uint32_t first_index = indices.size();
            uint32_t index_count = p.indices->count;

            primitive.first_vertex = first_vertex;
            primitive.first_index = first_index;
            primitive.index_count = index_count;
            primitive.material = cgltf_material_index(gltf_data, p.material);

            indices.resize(first_index + index_count);
            cgltf_accessor_unpack_indices(p.indices, indices.data() + first_index, sizeof(uint32_t), index_count);

            for (size_t k = 0; k < p.attributes_count; ++k)
            {
                const cgltf_attribute& a = p.attributes[k];
                switch (a.type)
                {
                case cgltf_attribute_type_position:
                {
                    assert(a.data->component_type == cgltf_component_type_r_32f);
                    assert(a.data->type == cgltf_type_vec3); // TODO: Handle other component types and types
                    position.resize(first_vertex + a.data->count);
                    cgltf_accessor_unpack_floats(a.data, (cgltf_float*)(position.data() + first_vertex), CGLTF_FLOAT_COUNT(a.data));
                    break;
                }
                case cgltf_attribute_type_normal:
                {
                    assert(a.data->component_type == cgltf_component_type_r_32f);
                    assert(a.data->type == cgltf_type_vec3); // TODO: Handle other component types and types
                    normal.resize(first_vertex + a.data->count);
                    cgltf_accessor_unpack_floats(a.data, (cgltf_float*)(normal.data() + first_vertex), CGLTF_FLOAT_COUNT(a.data));
                    break;
                }
                case cgltf_attribute_type_tangent:
                {
                    assert(a.data->component_type == cgltf_component_type_r_32f);
                    assert(a.data->type == cgltf_type_vec4); // TODO: Handle other component types and types
                    tangent.resize(first_vertex + a.data->count);
                    cgltf_accessor_unpack_floats(a.data, (cgltf_float*)(tangent.data() + first_vertex), CGLTF_FLOAT_COUNT(a.data));
                    break;
                }
                case cgltf_attribute_type_texcoord:
                {
                    cgltf_float* out = nullptr;
                    switch (a.index)
                    {
                    case 0:
                        texcoord0.resize(first_vertex + a.data->count);
                        out = (cgltf_float*)(texcoord0.data() + first_vertex);
                        break;
                    case 1:
                        texcoord1.resize(first_vertex + a.data->count);
                        out = (cgltf_float*)(texcoord1.data() + first_vertex);
                        break;
                    default:
                        LOG_WARNING("Unused texcoord index: %d", a.index);
                        break;
                    }
                    cgltf_accessor_unpack_floats(a.data, out, CGLTF_FLOAT_COUNT(a.data));

                    break;
                }
                default:
                    LOG_WARNING("Unused gltf attribute: %s", a.name);
                    break;
                }
            }
        }

        // Create GPU buffers
        { // Indices
            assert(!indices.empty());
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(indices);
            desc.usage_flags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = indices.data();
            mesh.indices = ctx.create_buffer(desc);
        }
        { // Position
            assert(!position.empty());
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(position);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = position.data();
            mesh.position = ctx.create_buffer(desc);
        }
        if (!normal.empty())
        { // Normal
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(normal);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = normal.data();
            mesh.normal = ctx.create_buffer(desc);
        }
        else
        {
            LOG_WARNING("Mesh has no normals!");
        }
        if (!tangent.empty())
        { // Tangent
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(tangent);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = tangent.data();
            mesh.tangent = ctx.create_buffer(desc);
        }
        else
        {
            LOG_WARNING("Mesh has no tangents!");
        }
        if (!texcoord0.empty())
        { // Texcoord0
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(texcoord0);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = texcoord0.data();
            mesh.texcoord0 = ctx.create_buffer(desc);
        }
        if (!texcoord1.empty())
        { // Texcoord1
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(texcoord1);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = texcoord1.data();
            mesh.texcoord1 = ctx.create_buffer(desc);
        }
    }

    std::vector<Material> materials;
    { // Load materials
        materials.resize(gltf_data->materials_count);
        
        for (size_t i = 0; i < gltf_data->materials_count; ++i)
        {
            const cgltf_material& mat = gltf_data->materials[i];

            Material& m = materials[i];
            assert(mat.has_pbr_metallic_roughness && "Unsupported material type!");

            m.basecolor_factor = glm::make_vec4(mat.pbr_metallic_roughness.base_color_factor);
            m.roughness_factor = mat.pbr_metallic_roughness.roughness_factor;
            m.metallic_factor = mat.pbr_metallic_roughness.metallic_factor;

            m.basecolor_texture = mat.pbr_metallic_roughness.base_color_texture.texture ? cgltf_texture_index(gltf_data, mat.pbr_metallic_roughness.base_color_texture.texture) : -1;
            m.metallic_roughness_texture = mat.pbr_metallic_roughness.metallic_roughness_texture.texture ? cgltf_texture_index(gltf_data, mat.pbr_metallic_roughness.metallic_roughness_texture.texture) : -1;
            m.normal_texture = mat.normal_texture.texture ? cgltf_texture_index(gltf_data, mat.normal_texture.texture) : -1;
        }
    }

    std::vector<Texture> textures;
    { // Load textures
        textures.resize(gltf_data->textures_count);
        

        for (size_t i = 0; i < gltf_data->textures_count; ++i)
        {
            const cgltf_texture& tex = gltf_data->textures[i];

            Texture& t = textures[i];

            if (tex.image->buffer_view)
            {
                int comp;
                t.source = stbi_load_from_memory((stbi_uc*)tex.image->buffer_view->data, tex.image->buffer_view->size, &t.width, &t.height, &comp, 4);
                assert(t.source);
            }
            else
            {
                std::string path = gltf_path;
                std::string uri = tex.image->uri;
                size_t last_slash = path.find_last_of('/');
                path = path.substr(0, last_slash + 1) + uri;
                int comp;
                t.source = stbi_load(path.c_str(), &t.width, &t.height, &comp, 4);
                assert(t.source);
            }


        }

        if (!ctx.create_textures(textures.data(), textures.size()))
        {
            exit(1);
        }
        
        std::vector<VkDescriptorImageInfo> image_info(textures.size());

        for (size_t i = 0; i < textures.size(); ++i)
        {
            VkDescriptorImageInfo& ii = image_info[i];
            ii.imageLayout = textures[i].layout;
            ii.imageView = textures[i].view;
        }
        VkWriteDescriptorSet desc_write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        desc_write.dstSet = ctx.bindless_descriptor_set;
        desc_write.dstBinding = 0;
        desc_write.dstArrayElement = 0;
        desc_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        desc_write.descriptorCount = textures.size();
        desc_write.pImageInfo = image_info.data();
        vkUpdateDescriptorSets(ctx.device, 1, &desc_write, 0, nullptr);
        /*
            VkStructureType                  sType;
    const void*                      pNext;
    VkDescriptorSet                  dstSet;
    uint32_t                         dstBinding;
    uint32_t                         dstArrayElement;
    uint32_t                         descriptorCount;
    VkDescriptorType                 descriptorType;
    const VkDescriptorImageInfo*     pImageInfo;
    const VkDescriptorBufferInfo*    pBufferInfo;
    const VkBufferView*              pTexelBufferView;*/

    }

    BufferDesc desc{};
    desc.size = sizeof(ShaderGlobals);
    desc.usage_flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    Buffer globals_buffer = ctx.create_buffer(desc);

    CameraState camera;
    float yaw = 0.0f;
    float pitch = 0.0f;

    uint64_t performance_frequency = SDL_GetPerformanceFrequency();
    double inv_pfreq = 1.0 / (double)performance_frequency;
    uint64_t start_tick = SDL_GetPerformanceCounter();
    uint64_t current_tick = start_tick;

    SDL_SetRelativeMouseMode(SDL_TRUE);

    float movement_speed = 1.0f;

    while (running)
    {
        VkCommandBuffer command_buffer = ctx.begin_frame();
        Texture& swapchain_texture = ctx.get_swapchain_texture();

        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            switch (e.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
            case SDL_KEYUP:
            {
                switch (e.key.keysym.scancode)
                {
                case SDL_SCANCODE_ESCAPE:
                    running = false;
                    break;
                }
            } break;
            case SDL_MOUSEWHEEL:
                movement_speed += e.wheel.y * 0.1f;
                break;
            default:
                break;
            }
        }

        movement_speed = std::max(movement_speed, 0.0f);

        int numkeys = 0;
        const uint8_t* keyboard = SDL_GetKeyboardState(&numkeys);

        int mousex, mousey;
        SDL_GetRelativeMouseState(&mousex, &mousey);

        constexpr float mouse_sensitivity = 0.1f;
        yaw -= mousex * mouse_sensitivity;
        pitch += mousey * mouse_sensitivity;

        if (glm::abs(yaw) > 180.0f) yaw -= glm::sign(yaw) * 360.0f;
        if (glm::abs(pitch) > 180.0f) pitch -= glm::sign(pitch) * 360.0f;

        glm::mat4 rotation = glm::yawPitchRoll(glm::radians(yaw), glm::radians(pitch), 0.0f);
        camera.forward = -rotation[2];

        glm::vec3 movement = glm::vec3(0.0f);
        if (keyboard[SDL_SCANCODE_W])       movement.z -= 1.0f;
        if (keyboard[SDL_SCANCODE_S])       movement.z += 1.0f;
        if (keyboard[SDL_SCANCODE_A])       movement.x -= 1.0f;
        if (keyboard[SDL_SCANCODE_D])       movement.x += 1.0f;
        if (keyboard[SDL_SCANCODE_SPACE])   movement.y += 1.0f;
        if (keyboard[SDL_SCANCODE_LCTRL])   movement.y -= 1.0f;

        if (glm::length(movement) != 0.0f) movement = glm::normalize(movement);

        uint64_t tick = SDL_GetPerformanceCounter();
        double delta_time = (tick - current_tick) * inv_pfreq;
        current_tick = tick;

        camera.position += glm::vec3(rotation * glm::vec4(movement, 0.0f)) * (float)delta_time * movement_speed;

        {
            ShaderGlobals globals{};
            globals.view = glm::lookAt(camera.position, camera.position + camera.forward, camera.up);
            globals.projection = glm::perspective(camera.fov, (float)window_width / (float)window_height, camera.znear, camera.zfar);
            globals.viewprojection = globals.projection * globals.view;

            void* mapped;
            vmaMapMemory(ctx.allocator, globals_buffer.allocation, &mapped);
            memcpy(mapped, &globals, sizeof(globals));
            vmaUnmapMemory(ctx.allocator, globals_buffer.allocation);
        }

        {
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                depth_texture.image,
                VK_IMAGE_ASPECT_DEPTH_BIT
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command_buffer, &dep_info);
        }

        VkRenderingAttachmentInfo color_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        color_info.imageView = swapchain_texture.view;
        color_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_info.clearValue.color = { 0.1f, 0.1f, 0.2f, 1.0f };

        VkRenderingAttachmentInfo depth_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depth_info.imageView = depth_texture.view;
        depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_info.clearValue.depthStencil.depth = 1.0f;

        VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        rendering_info.renderArea = { {0, 0}, {window_width, window_height} };
        rendering_info.layerCount = 1;
        rendering_info.viewMask = 0;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_info;
        rendering_info.pDepthAttachment = &depth_info;

        vkCmdBeginRendering(command_buffer, &rendering_info);

        VkRect2D scissor = { {0, 0}, {window_width, window_height} };
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        VkViewport viewport = {0.0f, (float)window_height, (float)window_width, -(float)window_height, 0.0f, 1.0f};
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 1, 1, &ctx.bindless_descriptor_set, 0, nullptr);

        auto render_node = [&](const cgltf_node* node)
            {
                if (node->mesh)
                {
                    size_t mesh_index = cgltf_mesh_index(gltf_data, node->mesh);
                    const Mesh& mesh = meshes[mesh_index];

                    struct
                    {
                        glm::mat4 model;
                        uint64_t position_buffer;
                        uint64_t normal_buffer;
                        uint64_t tangent_buffer;
                        uint64_t texcoord0_buffer;
                        uint64_t texcoord1_buffer;
                        int basecolor_texture;
                    } pc{};

                    cgltf_node_transform_world(node, glm::value_ptr(pc.model));
                    pc.position_buffer = ctx.buffer_device_address(mesh.position);
                    if (mesh.normal) pc.normal_buffer = ctx.buffer_device_address(mesh.normal);
                    if (mesh.tangent) pc.tangent_buffer = ctx.buffer_device_address(mesh.tangent);
                    if (mesh.texcoord0) pc.texcoord0_buffer = ctx.buffer_device_address(mesh.texcoord0);
                    if (mesh.texcoord1) pc.texcoord1_buffer = ctx.buffer_device_address(mesh.texcoord1);

                    vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                    for (const auto& primitive : mesh.primitives)
                    {
                        VkWriteDescriptorSet descriptors[2] = {};

                        VkDescriptorImageInfo iinfo0{};
                        iinfo0.sampler = sampler;
                        descriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        descriptors[0].dstSet = 0;
                        descriptors[0].dstBinding = 0;
                        descriptors[0].descriptorCount = 1;
                        descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                        descriptors[0].pImageInfo = &iinfo0;

                        VkDescriptorBufferInfo binfo0{};
                        binfo0.buffer = globals_buffer.buffer;
                        binfo0.offset = 0;
                        binfo0.range = VK_WHOLE_SIZE;
                        descriptors[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        descriptors[1].dstSet = 0;
                        descriptors[1].dstBinding = 1;
                        descriptors[1].descriptorCount = 1;
                        descriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                        descriptors[1].pBufferInfo = &binfo0;

                        pc.basecolor_texture = materials[primitive.material].basecolor_texture;
                        vkCmdPushConstants(command_buffer, pipeline_builder.pipeline_create_info.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                        vkCmdPushDescriptorSetKHR(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_builder.pipeline_create_info.layout, 0, std::size(descriptors), descriptors);
                        vkCmdDrawIndexed(command_buffer, primitive.index_count, 1, primitive.first_index, primitive.first_vertex, 0);
                    }
                }
            };

        const cgltf_scene* scene = gltf_data->scene;
        for (size_t i = 0; i < scene->nodes_count; ++i)
        {
            traverse_tree(scene->nodes[i], render_node);
        }

        vkCmdEndRendering(command_buffer);

        ctx.end_frame(command_buffer);
    }

    vkDeviceWaitIdle(ctx.device);
    vmaDestroyImage(ctx.allocator, depth_texture.image, depth_texture.allocation);
    vkDestroyImageView(ctx.device, depth_texture.view, nullptr);
    for (auto& m : meshes)
    {
        vmaDestroyBuffer(ctx.allocator, m.indices.buffer, m.indices.allocation);
        vmaDestroyBuffer(ctx.allocator, m.position.buffer, m.position.allocation);
        vmaDestroyBuffer(ctx.allocator, m.normal.buffer, m.normal.allocation);
        vmaDestroyBuffer(ctx.allocator, m.tangent.buffer, m.tangent.allocation);
        vmaDestroyBuffer(ctx.allocator, m.texcoord0.buffer, m.texcoord0.allocation);
        vmaDestroyBuffer(ctx.allocator, m.texcoord1.buffer, m.texcoord1.allocation);
    }
    vmaDestroyBuffer(ctx.allocator, globals_buffer.buffer, globals_buffer.allocation);
    vkDestroySampler(ctx.device, sampler, nullptr);
    for (auto& t : textures)
    {
        vkDestroyImageView(ctx.device, t.view, nullptr);
        vmaDestroyImage(ctx.allocator, t.image, t.allocation);
    }
    //vkDestroyDescriptorSetLayout(ctx.device, set_layout, nullptr);
    vkDestroyPipeline(ctx.device, pipeline.pipeline, nullptr);
    for (uint32_t i = 0; i < pipeline.descriptor_set_count; ++i)
    {
        if (pipeline.set_layouts[i] == ctx.bindless_descriptor_set_layout) continue;
        vkDestroyDescriptorSetLayout(ctx.device, pipeline.set_layouts[i], nullptr);
    }
    vkDestroyPipelineLayout(ctx.device, pipeline.layout, nullptr);

    ctx.shutdown();

    return 0;
}