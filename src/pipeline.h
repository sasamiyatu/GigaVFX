#pragma once

#include "defines.h"
#include "spirv_reflect.h"

struct DescriptorInfo
{
    union
    {
        VkDescriptorImageInfo image_info;
        VkDescriptorBufferInfo buffer_info;
    };

    inline DescriptorInfo(VkSampler sampler)
    {
        image_info = {};
        image_info.sampler = sampler;
    }

    inline DescriptorInfo(VkBuffer buffer, size_t offset = 0, size_t range = VK_WHOLE_SIZE)
    {
        buffer_info = {};
        buffer_info.buffer = buffer;
        buffer_info.offset = offset;
        buffer_info.range = range;
    }

    inline DescriptorInfo(VkImageView image_view, VkImageLayout layout)
    {
        image_info = {};
        image_info.imageView = image_view;
        image_info.imageLayout = layout;
    }
};

struct Pipeline
{
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layouts[4] = {};
    uint32_t descriptor_set_count = 0;
    VkDescriptorUpdateTemplate descriptor_update_template = VK_NULL_HANDLE;
};

struct GraphicsPipelineBuilder
{
    bool hot_reloadable = false;
    VkDevice device;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkGraphicsPipelineCreateInfo pipeline_create_info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

    static constexpr uint32_t max_shader_stages = 4;
    VkPipelineShaderStageCreateInfo shader_stage_create_info[max_shader_stages] = {};

    struct
    {
        uint32_t* spirv;
        uint32_t size;
        const char* filepath;
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

    static const uint32_t max_descriptor_set_layouts = 4;

    VkDescriptorSetLayout set_layouts[max_descriptor_set_layouts] = {};
    bool set_layout_passed_from_outside[max_descriptor_set_layouts] = { false };
    uint32_t descriptor_set_layout_count = 0;

    GraphicsPipelineBuilder(VkDevice dev, bool enable_shader_hot_reload);

    GraphicsPipelineBuilder& add_color_attachment(VkFormat format);
    GraphicsPipelineBuilder& set_depth_format(VkFormat format);
    GraphicsPipelineBuilder& set_depth_test(VkBool32 enabled);
    GraphicsPipelineBuilder& set_depth_write(VkBool32 enabled);
    GraphicsPipelineBuilder& set_depth_compare_op(VkCompareOp op);
    GraphicsPipelineBuilder& set_layout(VkPipelineLayout layout);
    GraphicsPipelineBuilder& set_cull_mode(VkCullModeFlagBits cull_mode);
    GraphicsPipelineBuilder& set_vertex_shader_filepath(const char* filepath);
    GraphicsPipelineBuilder& set_fragment_shader_filepath(const char* filepath);
    GraphicsPipelineBuilder& set_descriptor_set_layout(uint32_t set_index, VkDescriptorSetLayout layout);

    bool build(Pipeline* pipeline);
    void destroy_resources(Pipeline& pipeline);
};

struct ComputePipelineBuilder
{
    VkDevice device;
    bool hot_reloadable;
    VkComputePipelineCreateInfo create_info{};
    struct {
        uint32_t* spirv;
        uint32_t size;
        const char* filepath;
    } shader_source;
    static const uint32_t max_descriptor_set_layouts = 4;

    VkDescriptorSetLayout set_layouts[max_descriptor_set_layouts] = {};
    bool set_layout_passed_from_outside[max_descriptor_set_layouts] = { false };
    uint32_t descriptor_set_layout_count = 0;

    ComputePipelineBuilder& set_shader_filepath(const char* filepath);
    ComputePipelineBuilder(VkDevice device, bool enable_shader_hot_reload);

    bool build(Pipeline* pipeline);
    void destroy_resources(Pipeline& pipeline);
};

