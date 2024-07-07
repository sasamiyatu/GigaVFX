#include "pipeline.h"
#include "cmrc/cmrc.hpp"
CMRC_DECLARE(embedded_shaders);

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

GraphicsPipelineBuilder::GraphicsPipelineBuilder(VkDevice dev)
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

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_vertex_shader_spirv(uint32_t* data, size_t size)
{
    return add_shader_stage_spirv(data, size, VK_SHADER_STAGE_VERTEX_BIT, "vs_main");
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_fragment_shader_spirv(uint32_t* data, size_t size)
{
    return add_shader_stage_spirv(data, size, VK_SHADER_STAGE_FRAGMENT_BIT, "fs_main");
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_cull_mode(VkCullModeFlagBits cull_mode)
{
    rasterization_state.cullMode = cull_mode;

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_vertex_shader_filepath(const char* filepath)
{
    auto fs = cmrc::embedded_shaders::get_filesystem();
    std::string path = get_embedded_path(filepath, VK_SHADER_STAGE_VERTEX_BIT);
    auto file = fs.open(path);
    assert(file.size() % 4 == 0);
    return set_vertex_shader_spirv((uint32_t*)file.begin(), file.size());
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_fragment_shader_filepath(const char* filepath)
{
    auto fs = cmrc::embedded_shaders::get_filesystem();
    std::string path = get_embedded_path(filepath, VK_SHADER_STAGE_FRAGMENT_BIT);
    auto file = fs.open(path);
    assert(file.size() % 4 == 0);
    return set_fragment_shader_spirv((uint32_t*)file.begin(), file.size());
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::add_shader_stage_spirv(uint32_t* data, size_t size, VkShaderStageFlagBits shader_stage, const char* entry_point)
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

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_descriptor_set_layout(uint32_t set_index, VkDescriptorSetLayout layout)
{
    set_layouts[set_index] = layout;
    return *this;
}

Pipeline GraphicsPipelineBuilder::build()
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

ComputePipelineBuilder& ComputePipelineBuilder::set_shader_spirv(uint32_t* data, size_t size)
{
    VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = size;
    info.pCode = data;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &shader_module));

    VkPipelineShaderStageCreateInfo stage_info{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "cs_main";

    shader_source.spirv = data;
    shader_source.size = size;

    create_info.stage = stage_info;

    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::set_shader_filepath(const char* filepath)
{
    auto fs = cmrc::embedded_shaders::get_filesystem();
    std::string path = get_embedded_path(filepath, VK_SHADER_STAGE_COMPUTE_BIT);
    auto file = fs.open(path);
    assert(file.size() % 4 == 0);
    return set_shader_spirv((uint32_t*)file.begin(), file.size());
}

ComputePipelineBuilder::ComputePipelineBuilder(VkDevice device)
{
    create_info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    this->device = device;
}

Pipeline ComputePipelineBuilder::build()
{
    Pipeline pp{};

    if (create_info.layout == VK_NULL_HANDLE)
    {
        VkShaderStageFlags pc_stage_flags = 0;
        uint32_t pc_size = 0;

        SpvReflectShaderModule mod;
        SpvReflectResult result = spvReflectCreateShaderModule(shader_source.size, shader_source.spirv, &mod);
        assert(result == SPV_REFLECT_RESULT_SUCCESS);
        uint32_t descriptor_set_count = mod.descriptor_set_count;
        for (uint32_t j = 0; j < mod.push_constant_block_count; ++j)
        {
            pc_size += mod.push_constant_blocks[j].size;
            pc_stage_flags |= mod.shader_stage;
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings[4];
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
        range.size = pc_size;
        info.pPushConstantRanges = &range;
        info.pushConstantRangeCount = pc_size != 0 ? 1 : 0;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VK_CHECK(vkCreatePipelineLayout(device, &info, nullptr, &layout));
        create_info.layout = layout;
        pp.layout = layout;
        pp.descriptor_set_count = descriptor_set_count;
        memcpy(pp.set_layouts, set_layouts, sizeof(VkDescriptorSetLayout) * descriptor_set_count);

    }

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create_info, nullptr, &pp.pipeline));

    vkDestroyShaderModule(device, create_info.stage.module, nullptr);

    return pp;
}
