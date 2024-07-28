#include "pipeline.h"
#include "cmrc/cmrc.hpp"
#include "shaders.h"
#include <filesystem>
CMRC_DECLARE(embedded_shaders);
#include <optional>

    static uint64_t get_file_timestamp(const std::string& path)
    {
        std::filesystem::file_time_type ftime = std::filesystem::last_write_time(std::filesystem::path(path));
        return ftime.time_since_epoch().count();
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

GraphicsPipelineBuilder::GraphicsPipelineBuilder(VkDevice dev, bool enable_shader_hot_reload)
    : device(dev)
{
    // Set default values

    input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    dynamic_states[dynamic_state_count++] = VK_DYNAMIC_STATE_VIEWPORT;
    dynamic_states[dynamic_state_count++] = VK_DYNAMIC_STATE_SCISSOR;

    dynamic_state.dynamicStateCount = dynamic_state_count;
    dynamic_state.pDynamicStates = dynamic_states;

    rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization_state.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization_state.lineWidth = 1.0f;

    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::add_color_attachment(VkFormat format)
{
    color_blend_attachments[color_attachment_count].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_attachment_formats[color_attachment_count] = format;
    color_attachment_count++;

    color_blend_state.attachmentCount = color_attachment_count;
    rendering_create_info.colorAttachmentCount = color_attachment_count;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_depth_format(VkFormat format)
{
    rendering_create_info.depthAttachmentFormat = format;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_depth_test(VkBool32 enabled)
{
    depth_stencil_state.depthTestEnable = enabled;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_depth_write(VkBool32 enabled)
{
    depth_stencil_state.depthWriteEnable = enabled;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_depth_compare_op(VkCompareOp op)
{
    depth_stencil_state.depthCompareOp = op;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_layout(VkPipelineLayout layout)
{
    pipeline_create_info.layout = layout;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_cull_mode(VkCullModeFlagBits cull_mode)
{
    rasterization_state.cullMode = cull_mode;

    return *this;
}

void add_shader_stage(GraphicsPipelineBuilder& builder, VkShaderStageFlagBits shader_stage, const char* entry_point)
{
    uint32_t stage_count = builder.pipeline_create_info.stageCount;
    assert(stage_count < builder.max_shader_stages);

    VkPipelineShaderStageCreateInfo& stage_info = builder.shader_stage_create_info[stage_count];

    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = shader_stage;
    stage_info.module = VK_NULL_HANDLE;
    stage_info.pName = entry_point;

    builder.pipeline_create_info.stageCount++;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_vertex_shader_filepath(const char* filepath)
{
    shader_sources[pipeline_create_info.stageCount].filepath = filepath;
    add_shader_stage(*this, VK_SHADER_STAGE_VERTEX_BIT, "vs_main");
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_fragment_shader_filepath(const char* filepath)
{
    shader_sources[pipeline_create_info.stageCount].filepath = filepath;
    add_shader_stage(*this, VK_SHADER_STAGE_FRAGMENT_BIT, "fs_main");
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_descriptor_set_layout(uint32_t set_index, VkDescriptorSetLayout layout)
{
    set_layouts[set_index] = layout;
    set_layout_passed_from_outside[set_index] = true;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_view_mask(uint32_t mask)
{
    rendering_create_info.viewMask = mask;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_topology(VkPrimitiveTopology topology)
{
    input_assembly_state.topology = topology;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_blend_preset(BlendPreset preset)
{
    /*    VkBool32                 blendEnable;
    VkBlendFactor            srcColorBlendFactor;
    VkBlendFactor            dstColorBlendFactor;
    VkBlendOp                colorBlendOp;
    VkBlendFactor            srcAlphaBlendFactor;
    VkBlendFactor            dstAlphaBlendFactor;
    VkBlendOp                alphaBlendOp;
    VkColorComponentFlags    colorWriteMask;*/
    switch (preset)
    {
    case BlendPreset::ALPHA:
        color_blend_attachments[0].blendEnable = VK_TRUE;
        color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        color_blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
        color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
        break;
    case BlendPreset::ADDITIVE:
        color_blend_attachments[0].blendEnable = VK_TRUE;
        color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        color_blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
        color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
        break;
    case BlendPreset::NONE:
    default:
        break;
    }

    return *this;
}

bool GraphicsPipelineBuilder::build(Pipeline* out_pipeline)
{
    Pipeline& pp = *out_pipeline;

    Pipeline old_pipeline = *out_pipeline;

    hot_reloadable = true;
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
    dynamic_state.pDynamicStates = dynamic_states;

    color_blend_state.pAttachments = color_blend_attachments;

    rendering_create_info.pColorAttachmentFormats = color_attachment_formats;

    std::vector<VkDescriptorUpdateTemplateEntry> descriptor_template_entries;

    for (size_t i = 0; i < pipeline_create_info.stageCount; ++i)
    {
        shader_sources[i].spirv = Shaders::load_shader(shader_sources[i].filepath, pipeline_create_info.pStages[i].stage, &shader_sources[i].size);
        if (!shader_sources[i].spirv) return false;
        VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        info.codeSize = shader_sources[i].size;
        info.pCode = shader_sources[i].spirv;
        VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &shader_stage_create_info[i].module));
    }
    //if (pipeline_create_info.layout == VK_NULL_HANDLE)
    { // Create the layout
        std::vector<VkDescriptorSetLayoutBinding> bindings[4];
        uint32_t push_constant_size = 0;
        VkShaderStageFlags pc_stage_flags = 0;
        for (uint32_t i = 0; i < pipeline_create_info.stageCount; ++i)
        {
            SpvReflectShaderModule mod;
            SpvReflectResult result = spvReflectCreateShaderModule(shader_sources[i].size, shader_sources[i].spirv, &mod);
            assert(result == SPV_REFLECT_RESULT_SUCCESS);
            descriptor_set_layout_count = std::max(mod.descriptor_set_count, descriptor_set_layout_count);
            uint32_t pc_size = 0;
            for (uint32_t j = 0; j < mod.push_constant_block_count; ++j)
            {
                pc_size += mod.push_constant_blocks[j].size;
                pc_stage_flags |= mod.shader_stage;
            }

            push_constant_size = std::max(push_constant_size, pc_size);
            for (uint32_t j = 0; j < mod.descriptor_set_count; ++j)
            {
                if (set_layout_passed_from_outside[j]) continue;
                SpvReflectDescriptorSet descriptor_set = mod.descriptor_sets[j];
                bindings[j].resize(std::max(descriptor_set.binding_count, (uint32_t)bindings[j].size()));
                if (j == 0) descriptor_template_entries.resize(std::max((uint32_t)descriptor_template_entries.size(), descriptor_set.binding_count));

                for (uint32_t k = 0; k < descriptor_set.binding_count; ++k)
                {
                    SpvReflectDescriptorBinding* binding = descriptor_set.bindings[k];
                    VkDescriptorSetLayoutBinding& out_binding = bindings[j][binding->binding];
                    out_binding.binding = binding->binding;
                    out_binding.descriptorCount = binding->count;
                    out_binding.descriptorType = (VkDescriptorType)binding->descriptor_type;
                    out_binding.stageFlags |= mod.shader_stage;

                    if (j == 0) // Only create update template for set 0 for now
                    {
                        VkDescriptorUpdateTemplateEntry& entry = descriptor_template_entries[binding->binding];
                        entry.dstBinding = binding->binding;
                        entry.dstArrayElement = 0;
                        entry.descriptorCount = binding->count;
                        entry.descriptorType = (VkDescriptorType)binding->descriptor_type;
                        entry.offset = binding->binding * sizeof(DescriptorInfo);
                        entry.stride = sizeof(DescriptorInfo);
                    }
                }
            }
        }

        for (uint32_t i = 0; i < descriptor_set_layout_count; ++i)
        {
            if (set_layout_passed_from_outside[i]) continue;
            VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            info.bindingCount = bindings[i].size();
            info.pBindings = bindings[i].data();
            VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set_layouts[i]));
        }



        VkPipelineLayoutCreateInfo info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        info.setLayoutCount = descriptor_set_layout_count;
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
        pp.descriptor_set_count = descriptor_set_layout_count;
        memcpy(pp.set_layouts, set_layouts, sizeof(VkDescriptorSetLayout) * descriptor_set_layout_count);
    }

    VkDescriptorUpdateTemplateCreateInfo desc_template_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO };
    desc_template_info.descriptorUpdateEntryCount = descriptor_template_entries.size();
    desc_template_info.pDescriptorUpdateEntries = descriptor_template_entries.data();
    desc_template_info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    desc_template_info.descriptorSetLayout = pp.set_layouts[0];
    desc_template_info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    desc_template_info.pipelineLayout = pp.layout;
    desc_template_info.set = 0;

    VK_CHECK(vkCreateDescriptorUpdateTemplate(device, &desc_template_info, nullptr, &pp.descriptor_update_template));

    VK_CHECK(vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_create_info, nullptr, &pp.pipeline));

    destroy_resources(old_pipeline);

    for (uint32_t i = 0; i < pipeline_create_info.stageCount; ++i)
    {
        vkDestroyShaderModule(device, shader_stage_create_info[i].module, nullptr);
    }

    return true;
}

void GraphicsPipelineBuilder::destroy_resources(Pipeline& pipeline)
{
    for (uint32_t i = 0; i < 4; ++i)
    {
        if (!set_layout_passed_from_outside[i] && pipeline.set_layouts[i] != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, pipeline.set_layouts[i], nullptr);
            pipeline.set_layouts[i] = VK_NULL_HANDLE;
        }
    }

    if (pipeline.layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
        pipeline.layout = VK_NULL_HANDLE;
    }

    if (pipeline.pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }

    if (pipeline.descriptor_update_template != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorUpdateTemplate(device, pipeline.descriptor_update_template, nullptr);
        pipeline.descriptor_update_template = VK_NULL_HANDLE;
    }
}


ComputePipelineBuilder& ComputePipelineBuilder::set_shader_filepath(const char* filepath)
{
    shader_source.filepath = filepath;

    VkPipelineShaderStageCreateInfo stage_info{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.pName = "cs_main";

    create_info.stage = stage_info;

    return *this;
}

ComputePipelineBuilder::ComputePipelineBuilder(VkDevice device, bool enable_shader_hot_reload)
{
    hot_reloadable = true;
    create_info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    this->device = device;
}

bool ComputePipelineBuilder::build(Pipeline* out_pipeline)
{
    Pipeline& pp = *out_pipeline;
    Pipeline old_pipeline = *out_pipeline;

    std::vector<VkDescriptorUpdateTemplateEntry> descriptor_template_entries;

    {
        shader_source.spirv = Shaders::load_shader(shader_source.filepath, VK_SHADER_STAGE_COMPUTE_BIT, &shader_source.size);
        if (!shader_source.spirv) return false;

        VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        info.codeSize = shader_source.size;
        info.pCode = shader_source.spirv;
        VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &create_info.stage.module));
    }

    {
        VkShaderStageFlags pc_stage_flags = 0;
        uint32_t pc_size = 0;

        SpvReflectShaderModule mod;
        SpvReflectResult result = spvReflectCreateShaderModule(shader_source.size, shader_source.spirv, &mod);
        assert(result == SPV_REFLECT_RESULT_SUCCESS);
        descriptor_set_layout_count = mod.descriptor_set_count;
        for (uint32_t j = 0; j < mod.push_constant_block_count; ++j)
        {
            pc_size += mod.push_constant_blocks[j].size;
            pc_stage_flags |= mod.shader_stage;
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings[4];
        for (uint32_t j = 0; j < mod.descriptor_set_count; ++j)
        {
            if (set_layout_passed_from_outside[j]) continue;
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

                if (j == 0) // Only create update template for set 0 for now
                {
                    VkDescriptorUpdateTemplateEntry entry{};
                    entry.dstBinding = binding->binding;
                    entry.dstArrayElement = 0;
                    entry.descriptorCount = binding->count;
                    entry.descriptorType = (VkDescriptorType)binding->descriptor_type;
                    entry.offset = k * sizeof(DescriptorInfo);
                    entry.stride = sizeof(DescriptorInfo);
                    descriptor_template_entries.push_back(entry);
                }
            }
        }

        for (uint32_t i = 0; i < descriptor_set_layout_count; ++i)
        {
            if (set_layout_passed_from_outside[i]) continue;
            VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            info.bindingCount = bindings[i].size();
            info.pBindings = bindings[i].data();
            VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set_layouts[i]));
        }


        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipelineLayoutCreateInfo info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        info.setLayoutCount = descriptor_set_layout_count;
        info.pSetLayouts = set_layouts;
        VkPushConstantRange range{};
        range.stageFlags = pc_stage_flags;
        range.size = pc_size;
        info.pPushConstantRanges = &range;
        info.pushConstantRangeCount = pc_size != 0 ? 1 : 0;
        VK_CHECK(vkCreatePipelineLayout(device, &info, nullptr, &layout));
        create_info.layout = layout;
        pp.layout = layout;
        memcpy(pp.set_layouts, set_layouts, sizeof(VkDescriptorSetLayout) * descriptor_set_layout_count);
    }

    VkDescriptorUpdateTemplateCreateInfo desc_template_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO };
    desc_template_info.descriptorUpdateEntryCount = descriptor_template_entries.size();
    desc_template_info.pDescriptorUpdateEntries = descriptor_template_entries.data();
    desc_template_info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    desc_template_info.descriptorSetLayout = pp.set_layouts[0];
    desc_template_info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    desc_template_info.pipelineLayout = pp.layout;
    desc_template_info.set = 0;

    VK_CHECK(vkCreateDescriptorUpdateTemplate(device, &desc_template_info, nullptr, &pp.descriptor_update_template));

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create_info, nullptr, &pp.pipeline));

    destroy_resources(old_pipeline);

    vkDestroyShaderModule(device, create_info.stage.module, nullptr);

    return true;
}

void ComputePipelineBuilder::destroy_resources(Pipeline& pipeline)
{
    for (uint32_t i = 0; i < 4; ++i)
    {
        if (!set_layout_passed_from_outside[i] && pipeline.set_layouts[i] != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, pipeline.set_layouts[i], nullptr);
            pipeline.set_layouts[i] = VK_NULL_HANDLE;
        }
    }

    if (pipeline.layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
        pipeline.layout = VK_NULL_HANDLE;
    }

    if (pipeline.pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }

    if (pipeline.descriptor_update_template != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorUpdateTemplate(device, pipeline.descriptor_update_template, nullptr);
        pipeline.descriptor_update_template = VK_NULL_HANDLE;
    }
}

