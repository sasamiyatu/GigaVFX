#include "defines.h"
#include "VkBootstrap.h"
#include "vk_helpers.h"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Volk/volk.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "../shaders/shared.h"

#include "misc.h"
#include "pipeline.h"
#include "shaders.h"
#include "sdf.h"
#include "gmath.h"
#include "hot_reload.h"
#include "mesh.h"
#include "buffer.h"
#include "texture.h"
#include "gltf.h"
#include "graphics_context.h"
#include "particle_system.h"
#include "random.h"
#include "texture_catalog.h"    
#include "gpu_particles.h"
#include "radix_sort.h"
#include "camera.h"
#include "timer.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl2.h"
#include "imgui/imgui_impl_vulkan.h"

constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr uint32_t MAX_BINDLESS_RESOURCES = 1024;
constexpr VkFormat RENDER_TARGET_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr uint32_t DEPTH_TEXTURE_SIZE = 2048;

Context ctx;

static std::atomic<bool> needs_hot_reload = false;
static std::atomic<bool> hot_reload_watcher_should_quit = false;

template <typename F>
void traverse_tree(const cgltf_node* node, F&& f)
{
    f(node);
    for (size_t i = 0; i < node->children_count; ++i)
    {
        traverse_tree(node->children[i], std::forward<F>(f));
    }
}

bool init_imgui()
{
    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = ctx.instance;
    info.PhysicalDevice = ctx.physical_device;
    info.Device = ctx.device;
    info.QueueFamily = ctx.graphics_queue_family_index;
    info.Queue = ctx.graphics_queue;
    info.DescriptorPool = ctx.imgui_descriptor_pool;
    info.MinImageCount = ctx.swapchain.requested_min_image_count;
    info.ImageCount = ctx.swapchain.image_count;
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.UseDynamicRendering = true;
    VkFormat color_attachment_format = ctx.swapchain.image_format;
    info.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &color_attachment_format;
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImGui_ImplSDL2_InitForVulkan(ctx.window);

    ImGui_ImplVulkan_LoadFunctions([](const char* function_name, void*) { return vkGetInstanceProcAddr(ctx.instance, function_name); });

    bool success = ImGui_ImplVulkan_Init(&info);
    if (!success)
    {
        LOG_ERROR("Failed to initialize ImGui!");
        exit(1);
    }
    return success;
}

struct MeshInstance
{
    glm::mat4 transform;
    uint32_t mesh_index;
    uint32_t variant_index; 
};

namespace Input
{
#define MAX_KEYS 512
    uint8_t prev_keys[MAX_KEYS];
    uint8_t curr_keys[MAX_KEYS];

    void update()
    {
        memcpy(prev_keys, curr_keys, MAX_KEYS);
        int numkeys;
        const uint8_t* keys = SDL_GetKeyboardState(&numkeys);
        memcpy(curr_keys, keys, numkeys);
    }

    bool get_key_pressed(SDL_Scancode scancode)
    {
        return curr_keys[scancode] && !prev_keys[scancode];
    }

    bool get_key_released(SDL_Scancode scancode)
    {
        return !curr_keys[scancode] && prev_keys[scancode];
    }

    bool get_key_down(SDL_Scancode scancode)
    {
        return curr_keys[scancode];
    }
}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <path-to-glb-file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* gltf_path = argv[1];
    //const char* gltf_path = "D:/Projects/glTF-Sample-Models/2.0/Box/glTF/Box.gltf";
    cgltf_options opt{};
    cgltf_data* gltf_data = nullptr;
    cgltf_result res = cgltf_parse_file(&opt, gltf_path, &gltf_data);

    if (res != cgltf_result_success)
    {
        LOG_ERROR("Failed to load glTF!");
        exit(EXIT_FAILURE);
    }

    res = cgltf_load_buffers(&opt, gltf_data, gltf_path);
    if (res != cgltf_result_success)
    {
        LOG_ERROR("Failed to load buffers from glTF!");
        exit(EXIT_FAILURE);
    }

    SDF sdf{};
    if (!sdf_load_from_file(sdf, "data/dragon_small.sdf"))
    {
        LOG_ERROR("Failed to load SDF!");
        exit(EXIT_FAILURE);
    }

    ctx.init(WINDOW_WIDTH, WINDOW_HEIGHT);

    if (!sdf.init_texture(ctx))
    {
        LOG_ERROR("Failed to initialize SDF texture!");
        return(EXIT_FAILURE);
    }

    init_imgui();

    // Init shader compiler
    Shaders::init();

    VkSampler anisotropic_sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.maxLod = VK_LOD_CLAMP_NONE;
        info.anisotropyEnable = VK_TRUE;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.maxAnisotropy = ctx.physical_device.properties.limits.maxSamplerAnisotropy;
        VK_CHECK(vkCreateSampler(ctx.device, &info, nullptr, &anisotropic_sampler));
    }
    VkSampler bilinear_sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VK_CHECK(vkCreateSampler(ctx.device, &info, nullptr, &bilinear_sampler));
    }
	VkSampler point_sampler = VK_NULL_HANDLE;
	{
		VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
		info.magFilter = VK_FILTER_NEAREST;
		info.minFilter = VK_FILTER_NEAREST;
		info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		VK_CHECK(vkCreateSampler(ctx.device, &info, nullptr, &point_sampler));
	}
    VkSampler shadow_sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.compareEnable = VK_TRUE;
        info.compareOp = VK_COMPARE_OP_LESS;
        info.maxAnisotropy = ctx.physical_device.properties.limits.maxSamplerAnisotropy;
        VK_CHECK(vkCreateSampler(ctx.device, &info, nullptr, &shadow_sampler));
    }

    Texture depth_texture{};
    ctx.create_texture(depth_texture, WINDOW_WIDTH, WINDOW_HEIGHT, 1u, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, 
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    Texture shadowmap_texture{};
    ctx.create_texture(shadowmap_texture, DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE, 1u, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 4);

    Texture hdr_render_target{};
    ctx.create_texture(hdr_render_target, WINDOW_WIDTH, WINDOW_HEIGHT, 1, RENDER_TARGET_FORMAT, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    
    { 
        VkCommandBuffer cmd = ctx.allocate_and_begin_command_buffer();

        // Transition depth buffer layout
        // Nvidia hardware has no concept of image layouts, and that's what I'm using, so using VK_IMAGE_LAYOUT_GENERAL for now...
        VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            0,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            depth_texture.image,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );

        VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep_info.imageMemoryBarrierCount = 1;
        dep_info.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep_info);

        ctx.end_command_buffer_submit_and_free(cmd);
    }

    GraphicsPipelineAsset* depth_prepass = nullptr;
    { // Depth prepass pipeline
        GraphicsPipelineBuilder pipeline_builder(ctx.device, true);
        pipeline_builder.set_vertex_shader_filepath("depth_prepass.hlsl")
            .set_fragment_shader_filepath("depth_prepass.hlsl")
            .set_cull_mode(VK_CULL_MODE_NONE)
            .set_depth_format(VK_FORMAT_D32_SFLOAT)
            .set_depth_test(VK_TRUE)
            .set_depth_write(VK_TRUE)
            .set_depth_compare_op(VK_COMPARE_OP_LESS)
            .set_descriptor_set_layout(1, ctx.bindless_descriptor_set_layout);

        depth_prepass = new GraphicsPipelineAsset(pipeline_builder);
        AssetCatalog::register_asset(depth_prepass);
    }

    // Variant with discard based on noise
	GraphicsPipelineAsset* depth_prepass_disintegrate;
	{
		GraphicsPipelineBuilder pipeline_builder(ctx.device, true);
        ShaderSource vertex_source("depth_prepass.hlsl", "vs_main");
        vertex_source.add_specialization_constant(1, true);
        ShaderSource fragment_source("depth_prepass.hlsl", "fs_main");
        fragment_source.add_specialization_constant(1, true);
		pipeline_builder
            .set_vertex_shader_source(vertex_source)
			.set_fragment_shader_source(fragment_source)
			.set_cull_mode(VK_CULL_MODE_NONE)
			.set_depth_format(VK_FORMAT_D32_SFLOAT)
			.set_depth_test(VK_TRUE)
			.set_depth_write(VK_TRUE)
			.set_depth_compare_op(VK_COMPARE_OP_LESS)
            .set_descriptor_set_layout(1, ctx.bindless_descriptor_set_layout);
		depth_prepass_disintegrate = new GraphicsPipelineAsset(pipeline_builder);
		AssetCatalog::register_asset(depth_prepass_disintegrate);
	}

    GraphicsPipelineBuilder pipeline_builder(ctx.device, true);
    pipeline_builder
        .set_vertex_shader_filepath("forward.hlsl")
        .set_fragment_shader_filepath("forward.hlsl")
        .add_color_attachment(RENDER_TARGET_FORMAT)
        .set_cull_mode(VK_CULL_MODE_NONE)
        .set_depth_format(VK_FORMAT_D32_SFLOAT)
        .set_depth_test(VK_TRUE)
        .set_depth_write(VK_FALSE)
        .set_depth_compare_op(VK_COMPARE_OP_EQUAL)
        .set_descriptor_set_layout(1, ctx.bindless_descriptor_set_layout);

    GraphicsPipelineAsset* pipeline = new GraphicsPipelineAsset(pipeline_builder);
    AssetCatalog::register_asset(pipeline);


    GraphicsPipelineBuilder shadowmap_builder(ctx.device, true);
    shadowmap_builder
        .set_vertex_shader_filepath("shadowmap.hlsl")
        .set_fragment_shader_filepath("shadowmap.hlsl")
        .set_cull_mode(VK_CULL_MODE_NONE)
        .set_depth_format(VK_FORMAT_D32_SFLOAT)
        .set_depth_test(VK_TRUE)
        .set_depth_write(VK_TRUE)
        .set_depth_compare_op(VK_COMPARE_OP_LESS)
        .set_view_mask(0b1111)
        .set_descriptor_set_layout(1, ctx.bindless_descriptor_set_layout);

    GraphicsPipelineAsset* shadowmap_pipeline = new GraphicsPipelineAsset(shadowmap_builder);
    AssetCatalog::register_asset(shadowmap_pipeline);

    GraphicsPipelineAsset* shadowmap_disintegrate_pipeline = nullptr;
    {
        GraphicsPipelineBuilder builder(ctx.device, true);
		ShaderSource fragment_source("shadowmap.hlsl", "fs_main");
		fragment_source.add_specialization_constant(1, true);
        builder
            .set_vertex_shader_filepath("shadowmap.hlsl")
			.set_fragment_shader_source(fragment_source)
            .set_cull_mode(VK_CULL_MODE_NONE)
            .set_depth_format(VK_FORMAT_D32_SFLOAT)
            .set_depth_test(VK_TRUE)
            .set_depth_write(VK_TRUE)
            .set_depth_compare_op(VK_COMPARE_OP_LESS)
            .set_view_mask(0b1111)
            .set_descriptor_set_layout(1, ctx.bindless_descriptor_set_layout);
        shadowmap_disintegrate_pipeline = new GraphicsPipelineAsset(builder);
		AssetCatalog::register_asset(shadowmap_disintegrate_pipeline);
    }

    ComputePipelineBuilder compute_builder(ctx.device, true);
    compute_builder
        .set_shader_filepath("procedural_sky.hlsl");
    ComputePipelineAsset* procedural_skybox_pipeline = new ComputePipelineAsset(compute_builder);
    AssetCatalog::register_asset(procedural_skybox_pipeline);

    ComputePipelineAsset* tonemap_pipeline = nullptr;
    {
        ComputePipelineBuilder compute_builder(ctx.device, true);
        compute_builder.set_shader_filepath("tonemap.hlsl");
        tonemap_pipeline = new ComputePipelineAsset(compute_builder);
        AssetCatalog::register_asset(tonemap_pipeline);
    }

    std::vector<Mesh> meshes;
    meshes.resize(gltf_data->meshes_count);
    load_meshes(ctx, gltf_data, meshes.data(), meshes.size());

    std::vector<Material> materials;
    materials.resize(gltf_data->materials_count);
    load_materials(ctx, gltf_data, materials.data(), materials.size());
    Buffer materials_buffer;
    {
        BufferDesc desc{};
        desc.size = sizeof(Material) * materials.size();
        desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.data = materials.data();
        materials_buffer = ctx.create_buffer(desc);
    }

    std::vector<Texture> textures(gltf_data->textures_count);
    load_textures(ctx, gltf_data, gltf_path, textures.data(), textures.size());

    BufferDesc desc{};
    desc.size = sizeof(ShaderGlobals);
    desc.usage_flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    //desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    //Buffer globals_buffer = ctx.create_buffer(desc);
    GPUBuffer globals_buffer = ctx.create_gpu_buffer(desc);

    CameraState camera;
    float yaw = 0.0f;
    float pitch = 0.0f;

    uint64_t performance_frequency = SDL_GetPerformanceFrequency();
    double inv_pfreq = 1.0 / (double)performance_frequency;
    uint64_t start_tick = SDL_GetPerformanceCounter();
    uint64_t current_tick = start_tick;

    float movement_speed = 1.0f;

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImGuiIO& io = ImGui::GetIO();

    float slider[4] = {};

    TextureCatalog texture_catalog;
    texture_catalog.init(&ctx, "data/textures/");

    ParticleRenderer particle_renderer;
    particle_renderer.init(&ctx, globals_buffer, RENDER_TARGET_FORMAT);
    particle_renderer.texture_catalog = &texture_catalog;

    ParticleSystemManager particle_system_manager;
    particle_system_manager.init(&particle_renderer);

    std::vector<IConfigUI*> config_uis;
    constexpr uint32_t particle_capacity = 1048576;
    GPUParticleSystem smoke_system;
    smoke_system.init(&ctx, globals_buffer, RENDER_TARGET_FORMAT, particle_capacity, shadowmap_texture, 1,
        {"gpu_particles.hlsl", "cs_emit_particles"}, {"gpu_particles.hlsl", "cs_simulate_particles"});
    smoke_system.set_position(glm::vec3(0.0f, 0.0f, 0.0f));
    config_uis.push_back(&smoke_system);

    GPUSurfaceFlowSystem flow2;
    flow2.init(&ctx, globals_buffer, RENDER_TARGET_FORMAT, 30000,
        { "surface_flow.hlsl", "emit" }, { "surface_flow.hlsl", "simulate" }, &sdf, false);
    flow2.set_position(glm::vec3(-2.0, 0.0f, 0.0f));
    config_uis.push_back(&flow2);

    TrailBlazerSystem trail_blazer;
    trail_blazer.sdf = &sdf;
    trail_blazer.init(&ctx, globals_buffer, RENDER_TARGET_FORMAT);
    config_uis.push_back(&trail_blazer);

	ParticleManagerSimple particle_manager;
    particle_manager.init(&ctx, globals_buffer, RENDER_TARGET_FORMAT);
    
    {
        ParticleSystemSimple::Config config{};
        config.emit_and_simulate_file = "particle_simple.hlsli";
        config.particle_capacity = 32678;
        config.spawn_rate = 1000.0f;
        config.name = "Particle Simple";
        particle_manager.add_system(config);
    }

    {
        ParticleSystemSimple::Config config{};
        config.emit_and_simulate_file = "particle_simple2.hlsli";
        config.particle_capacity = 32678;
        config.spawn_rate = 1000.0f;
        config.name = "Particle Simple2";
		//particle_manager.add_system(config);
    }

    Buffer mesh_disintegrate_spawn_positions;

    {
        BufferDesc desc{};
        desc.size = sizeof(glm::vec3) * WINDOW_WIDTH * WINDOW_HEIGHT;
        desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        mesh_disintegrate_spawn_positions = ctx.create_buffer(desc);
    }

	ParticleSystemSimple* disintegrator_system = nullptr;
    {
        ParticleSystemSimple::Config config{};
        config.emit_and_simulate_file = "mesh_disintegrate.hlsli";
        config.particle_capacity = 262144;
        config.name = "MeshDisintegrate";
        config.emit_indirect_dispatch_handled_externally = true;
        config.additional_descriptors = {
            DescriptorInfo(mesh_disintegrate_spawn_positions.buffer),
            DescriptorInfo(depth_texture.view, VK_IMAGE_LAYOUT_GENERAL),
            DescriptorInfo(point_sampler)
        };
        disintegrator_system = particle_manager.add_system(config);
    }

	for (auto& s : particle_manager.systems) config_uis.push_back(s);

    std::vector<MeshInstance> mesh_draws;

    // Test acceleration structure
    ComputePipelineBuilder builder(ctx.device, true);
    builder.set_shader_filepath("test_acceleration_structure.hlsl", "test_acceleration_structure");
    ComputePipelineAsset* test_pipeline = new ComputePipelineAsset(builder);
    AssetCatalog::register_asset(test_pipeline);

    // Test sdf
    ComputePipelineAsset* sdf_test = nullptr;
    {
        ComputePipelineBuilder builder(ctx.device, true);
        builder.set_shader_filepath("sdf_test.hlsl", "test_sdf");
        sdf_test = new ComputePipelineAsset(builder);
        AssetCatalog::register_asset(sdf_test);
    }

    // Separate thread for checking need for hot reload
    std::thread hot_reload_watcher([]() {
        while (!hot_reload_watcher_should_quit)
        {
            if (!needs_hot_reload)
            {
                needs_hot_reload = AssetCatalog::check_for_dirty_assets();
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    glm::vec3 sundir = glm::normalize(glm::vec3(1.0f));
    bool running = true;
    bool texture_catalog_open = true;
    uint32_t frame_index = 0;
    bool show_imgui_demo = false;
    double cpu_time_ms = 0.0;

    glm::vec3 smoke_dir = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 smoke_origin = glm::vec3(0.0f);
    {
        auto print_hierarchy = [&](const cgltf_node* node)
            {
                if (strcmp(node->name, "smoke_origin") == 0)
                {
                    printf("%s\n", node->name);
                    glm::mat4 transform;
                    cgltf_node_transform_world(node, glm::value_ptr(transform));
                    smoke_dir = glm::normalize(glm::vec3(transform[2]));
                    smoke_origin = glm::vec3(transform[3]);
                }
            };

        const cgltf_scene* scene = gltf_data->scene;
        for (size_t i = 0; i < scene->nodes_count; ++i)
        {
            traverse_tree(scene->nodes[i], print_hierarchy);
        }
    }

    smoke_system.smoke_origin = smoke_origin;
    smoke_system.smoke_dir = smoke_dir;

    while (running)
    {
        Timer timer;
        timer.tick();
        VkCommandBuffer command_buffer = ctx.begin_frame();
        Texture& swapchain_texture = ctx.get_swapchain_texture();

        VkHelpers::begin_label(command_buffer, "Frame start", glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        { // Frame stats
            ImGui::Begin("GPU Particle System");
            ImGui::Text("GPU frame time: %f ms", ctx.smoothed_frame_time_ns * 1e-6f);
            ImGui::Text("CPU frame time: %f ms", cpu_time_ms);
            ImGui::Text("Particle simulate: %f us", smoke_system.performance_timings.simulate_total * 1e-3f);
            ImGui::Text("Particle render: %f us", smoke_system.performance_timings.render_total * 1e-3f);
            ImGui::Separator();
            static int selected_system = 0;
            if (ImGui::BeginCombo("Particle system", config_uis[selected_system]->get_display_name()))
            {
                for (int i = 0; i < config_uis.size(); ++i)
                {
                    if (ImGui::Selectable(config_uis[i]->get_display_name(), selected_system == i))
                    {
                        selected_system = i;
                    }
                }

                ImGui::EndCombo();
            }
            config_uis[selected_system]->draw_config_ui();
            //smoke_system.draw_config_ui();
            ImGui::End();
        }

        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            ImGui_ImplSDL2_ProcessEvent(&e);
            switch (e.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
            {
                if (io.WantCaptureKeyboard) break;

                switch (e.key.keysym.scancode)
                {
                case SDL_SCANCODE_ESCAPE:
                    running = false;
                    break;
                case SDL_SCANCODE_F5:
                    AssetCatalog::force_reload_all();
                    break;
                case SDL_SCANCODE_F10:
                    show_imgui_demo = !show_imgui_demo;
                    break;
                default:
                    break;
                }
            } break;
            case SDL_KEYUP:
            {

            } break;
            case SDL_MOUSEWHEEL:
                if (io.WantCaptureMouse) break;
                movement_speed += e.wheel.y * 0.1f;
                break;
            case SDL_MOUSEBUTTONDOWN:
            case SDL_MOUSEBUTTONUP:
                if (io.WantCaptureMouse) break;
                if (e.button.button == SDL_BUTTON_LEFT)
                    SDL_SetRelativeMouseMode(e.type == SDL_MOUSEBUTTONDOWN ? SDL_TRUE : SDL_FALSE);
                break;
            default:
                break;
            }
        }

        Input::update();
        if (Input::get_key_pressed(SDL_SCANCODE_F1))
        {
            texture_catalog_open = !texture_catalog_open;
        }

        if (texture_catalog_open)
        {
            //texture_catalog.draw_ui(&texture_catalog_open);
        }

        if (show_imgui_demo)
            ImGui::ShowDemoWindow(&show_imgui_demo);

        //particle_system_manager.draw_ui();
        //gpu_particle_system.draw_stats_overlay();

        movement_speed = std::max(movement_speed, 0.0f);

        int numkeys = 0;
        const uint8_t* keyboard = SDL_GetKeyboardState(&numkeys);

        int mousex, mousey;
        uint32_t mouse_buttons = SDL_GetRelativeMouseState(&mousex, &mousey);

        constexpr float mouse_sensitivity = 0.1f;
        if (!io.WantCaptureMouse && (mouse_buttons & SDL_BUTTON_LMASK))
        {
            yaw -= mousex * mouse_sensitivity;
            pitch += mousey * mouse_sensitivity;

            if (glm::abs(yaw) > 180.0f) yaw -= glm::sign(yaw) * 360.0f;
            if (glm::abs(pitch) > 180.0f) pitch -= glm::sign(pitch) * 360.0f;
        }

        glm::mat4 rotation = glm::yawPitchRoll(glm::radians(yaw), glm::radians(pitch), 0.0f);
        camera.forward = -rotation[2];

        glm::vec3 movement = glm::vec3(0.0f);
        if (!io.WantCaptureKeyboard)
        {
            if (keyboard[SDL_SCANCODE_W])       movement.z -= 1.0f;
            if (keyboard[SDL_SCANCODE_S])       movement.z += 1.0f;
            if (keyboard[SDL_SCANCODE_A])       movement.x -= 1.0f;
            if (keyboard[SDL_SCANCODE_D])       movement.x += 1.0f;
            if (keyboard[SDL_SCANCODE_SPACE])   movement.y += 1.0f;
            if (keyboard[SDL_SCANCODE_LCTRL])   movement.y -= 1.0f;
        }

        if (glm::length(movement) != 0.0f) movement = glm::normalize(movement);

        uint64_t tick = SDL_GetPerformanceCounter();
        double delta_time = (tick - current_tick) * inv_pfreq;
        double elapsed_time = (tick - start_tick) * inv_pfreq;
        current_tick = tick;

        const float disintegrate_alpha_reference = glm::fract(elapsed_time * 0.1f);
		const float disintegrate_prev_alpha_reference = glm::fract((elapsed_time - delta_time) * 0.1f);

        particle_system_manager.update((float)delta_time);

        if (needs_hot_reload)
        {
            VK_CHECK(vkDeviceWaitIdle(ctx.device));
            while (!AssetCatalog::reload_dirty_assets())
            {
                SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Shader compilation error", "Shader compilation failed!\nRetry?", ctx.window);
            }

            needs_hot_reload = false;
        }

        camera.position += glm::vec3(rotation * glm::vec4(movement, 0.0f)) * (float)delta_time * movement_speed;

        { // Collect mesh instances from scene
            mesh_draws.clear();
            auto get_meshes = [&](const cgltf_node* node)
                {
                    if (node->mesh)
                    {
                        MeshInstance& mi = mesh_draws.emplace_back();
                        mi.mesh_index = (uint32_t)cgltf_mesh_index(gltf_data, node->mesh);
						if (node->name && strcmp(node->name, "dragon_mat") == 0)
						{
							mi.variant_index = 1;
						}
                        else
                        {
                            mi.variant_index = 0;
                        }
                        cgltf_node_transform_world(node, glm::value_ptr(mi.transform));
                    }
                };

            const cgltf_scene* scene = gltf_data->scene;
            for (size_t i = 0; i < scene->nodes_count; ++i)
            {
                traverse_tree(scene->nodes[i], get_meshes);
            }

			std::sort(mesh_draws.begin(), mesh_draws.end(), [](const MeshInstance& a, const MeshInstance& b) { return a.variant_index < b.variant_index; });
        }

        glm::mat4 shadow_projs[4];
        glm::mat4 shadow_views[4];
        glm::mat4 shadow_view_projs[4];
        { // Update global uniform buffer
            glm::ivec2 resolution;
			SDL_GetWindowSizeInPixels(ctx.window, &resolution.x, &resolution.y);
            ShaderGlobals globals{};
            globals.view = glm::lookAt(camera.position, camera.position + camera.forward, camera.up);
            globals.view_inverse = glm::inverse(globals.view);
            globals.projection = glm::perspective(camera.fov, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, camera.znear, camera.zfar);
            globals.projection_inverse = glm::inverse(globals.projection);
            globals.viewprojection = globals.projection * globals.view;
            globals.viewprojection_inverse = glm::inverse(globals.viewprojection);
            globals.camera_pos = glm::vec4(camera.position, 1.0f);
            globals.sun_direction = glm::vec4(sundir, 1.0f);
            globals.sun_color_and_intensity = glm::vec4(1.0f);
            globals.resolution = glm::vec2(resolution);
            globals.frame_index = frame_index;
            globals.time = (float)elapsed_time;

            glm::vec4 origin_shift[4];
            float max_distance = 100.0f;

            float distance_thresholds[4] = { 0.0f, 5.0f, 15.0f, 45.0f };
            memcpy(glm::value_ptr(globals.shadow_cascade_thresholds), distance_thresholds, sizeof(distance_thresholds));
            for (int i = 0; i < 4; ++i)
            {
                float near_plane = std::max(distance_thresholds[i], 0.01f);
                float far_plane = i < 3 ? distance_thresholds[i + 1] : max_distance;
                glm::mat4 proj = glm::perspective(camera.fov, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, near_plane, far_plane);
                Sphere frustum_bounding_sphere = get_frustum_bounding_sphere(proj);
                float r = frustum_bounding_sphere.radius;
                glm::mat4 shadow_proj = glm::ortho(-r, r, -r, r, 0.1f, 2.0f * r);
                shadow_projs[i] = shadow_proj;

                glm::vec3 cascade_center = glm::vec3(globals.view_inverse * glm::vec4(frustum_bounding_sphere.center, 1.0f));
                glm::mat4 shadow_view = glm::lookAt(cascade_center + sundir * r, cascade_center, glm::vec3(0.0f, 1.0f, 0.0f));
                shadow_views[i] = shadow_view;

                shadow_view_projs[i] = shadow_proj * shadow_view;

                globals.shadow_view[i] = shadow_view;
                globals.shadow_projection[i] = shadow_proj;
                globals.shadow_view_projection[i] = shadow_proj * shadow_view;

                float znear = shadow_proj[3][2] / shadow_proj[2][2];
                float zfar_minus_znear = -1.0f / shadow_proj[2][2];
                globals.shadow_projection_info[i] = glm::vec4(zfar_minus_znear, znear, 0.0f, 0.0f);

                // Snap to pixel center to avoid shimmering
                glm::vec4 shadow_origin = shadow_view_projs[i] * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                shadow_origin /= shadow_origin.w;
                float step = 1.0f / ((float)DEPTH_TEXTURE_SIZE * 0.5f);
                glm::vec2 rounded = glm::round(glm::vec2(shadow_origin) / step) * step;

                glm::mat4 translate = glm::translate(glm::mat4(1.0f), glm::vec3(rounded.x - shadow_origin.x, rounded.y - shadow_origin.y, 0.0f));
                shadow_projs[i] = globals.shadow_projection[i] = translate * shadow_proj;
                globals.shadow_view_projection[i] = translate * globals.shadow_view_projection[i];
            }

            void* mapped;
            ctx.map_buffer(globals_buffer, &mapped);
            memcpy(mapped, &globals, sizeof(globals));
            ctx.unmap_buffer(globals_buffer);
            ctx.upload_buffer(globals_buffer, command_buffer);
            VkHelpers::memory_barrier(command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0);
        }

        VkHelpers::end_label(command_buffer);

        smoke_system.simulate(command_buffer, (float)delta_time, camera, shadow_views[1], shadow_projs[1]);
        flow2.simulate(command_buffer, (float)delta_time);
        trail_blazer.simulate(command_buffer, (float)delta_time);

		particle_manager.update_systems(command_buffer, (float)delta_time);

        {
            // Clear dispatch count because depth prepass will write it
            vkCmdFillBuffer(command_buffer, disintegrator_system->emit_indirect_dispatch_buffer, 0, sizeof(uint32_t), 0);
			VkHelpers::memory_barrier(command_buffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }

        { // Transition render target
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                hdr_render_target.image,
                VK_IMAGE_ASPECT_COLOR_BIT
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command_buffer, &dep_info);
        }

        { // Procedural sky box
            VkHelpers::begin_label(command_buffer, "Procedural sky box", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
           
            DescriptorInfo descriptor_info[] = {
                DescriptorInfo(globals_buffer),
                DescriptorInfo(hdr_render_target.view, VK_IMAGE_LAYOUT_GENERAL)
            };

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, procedural_skybox_pipeline->pipeline.descriptor_update_template, procedural_skybox_pipeline->pipeline.layout, 0, descriptor_info);
            
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, procedural_skybox_pipeline->pipeline.pipeline);

            uint32_t dispatch_x = get_golden_dispatch_size(WINDOW_WIDTH);
            uint32_t dispatch_y = get_golden_dispatch_size(WINDOW_HEIGHT);
            vkCmdDispatch(command_buffer, dispatch_x, dispatch_y, 1);

            VkHelpers::end_label(command_buffer);
        }

        { // Shadowmap
            VkHelpers::begin_label(command_buffer, "Cascaded shadow map", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
            VkImageMemoryBarrier2 barrier = VkHelpers::image_memory_barrier2(
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                shadowmap_texture.image,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                0,
                1,
                0,
                4
            );

            VkDependencyInfo dep_info{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command_buffer, &dep_info);

            VkRenderingAttachmentInfo depth_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            depth_info.imageView = shadowmap_texture.view;
            depth_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            depth_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_info.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            rendering_info.renderArea = { {0, 0}, {DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE} };
            rendering_info.layerCount = 4;
            rendering_info.viewMask = 0b1111;
            rendering_info.pDepthAttachment = &depth_info;

            VkRect2D scissor = { {0, 0}, {DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE} };
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            VkViewport viewport = { 0.0f, (float)DEPTH_TEXTURE_SIZE, (float)DEPTH_TEXTURE_SIZE, -(float)DEPTH_TEXTURE_SIZE, 0.0f, 1.0f };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            vkCmdBeginRendering(command_buffer, &rendering_info);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowmap_pipeline->pipeline.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowmap_pipeline->pipeline.layout, 1, 1, &ctx.bindless_descriptor_set, 0, nullptr);

            DescriptorInfo descriptor_info[] = {
                DescriptorInfo(anisotropic_sampler),
                DescriptorInfo(globals_buffer),
                DescriptorInfo(materials_buffer.buffer)
            };

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, shadowmap_pipeline->pipeline.descriptor_update_template, shadowmap_pipeline->pipeline.layout, 0, descriptor_info);

            for (const auto& mi : mesh_draws)
            {
				if (mi.variant_index == 1) vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowmap_disintegrate_pipeline->pipeline.pipeline);

                const Mesh& mesh = meshes[mi.mesh_index];

                DepthPrepassPushConstants pc{};
                static_assert(sizeof(pc) <= 128);

                pc.model = mi.transform;
                pc.position_buffer = ctx.buffer_device_address(mesh.position);
				pc.alpha_reference = disintegrate_alpha_reference;
                pc.prev_alpha_reference = disintegrate_prev_alpha_reference;
                if (mesh.texcoord0) pc.texcoord0_buffer = ctx.buffer_device_address(mesh.texcoord0);

                vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                for (const auto& primitive : mesh.primitives)
                {
                    pc.noise_texture_index = materials[primitive.material].basecolor_texture;
                    if (mi.variant_index == 1)
                    {
                        assert(pc.noise_texture_index >= 0);
                    }

                    vkCmdPushConstants(command_buffer, shadowmap_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                    vkCmdDrawIndexed(command_buffer, primitive.index_count, 1, primitive.first_index, primitive.first_vertex, 0);
                }
            }

            vkCmdEndRendering(command_buffer);

            VkHelpers::end_label(command_buffer);
        }

        { // Depth prepass
            VkHelpers::begin_label(command_buffer, "Depth prepass", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

            VkRenderingAttachmentInfo depth_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            depth_info.imageView = depth_texture.view;
            depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_info.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            rendering_info.renderArea = { {0, 0}, {WINDOW_WIDTH, WINDOW_HEIGHT} };
            rendering_info.layerCount = 1;
            rendering_info.viewMask = 0;
            rendering_info.pDepthAttachment = &depth_info;

            vkCmdBeginRendering(command_buffer, &rendering_info);

            VkRect2D scissor = { {0, 0}, {WINDOW_WIDTH, WINDOW_HEIGHT} };
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            VkViewport viewport = { 0.0f, (float)WINDOW_HEIGHT, (float)WINDOW_WIDTH, -(float)WINDOW_HEIGHT, 0.0f, 1.0f };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_prepass->pipeline.pipeline);
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_prepass->pipeline.layout, 1, 1, &ctx.bindless_descriptor_set, 0, nullptr);
            DescriptorInfo descriptor_info[] = {
                DescriptorInfo(globals_buffer),
                DescriptorInfo(bilinear_sampler),
                DescriptorInfo(disintegrator_system->particle_system_state[0]),
                DescriptorInfo(disintegrator_system->emit_indirect_dispatch_buffer),
				DescriptorInfo(mesh_disintegrate_spawn_positions.buffer),
            };

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, depth_prepass->pipeline.descriptor_update_template, 
                depth_prepass->pipeline.layout, 0, descriptor_info);

            for (const auto& mi : mesh_draws)
            {
                const Mesh& mesh = meshes[mi.mesh_index];
                
                if (mi.variant_index == 1) vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_prepass_disintegrate->pipeline.pipeline);

                DepthPrepassPushConstants pc{};
                static_assert(sizeof(pc) <= 128);

                pc.model = mi.transform;
                pc.position_buffer = ctx.buffer_device_address(mesh.position);
                pc.alpha_reference = disintegrate_alpha_reference;
                pc.prev_alpha_reference = disintegrate_prev_alpha_reference;
                if (mesh.texcoord0) pc.texcoord0_buffer = ctx.buffer_device_address(mesh.texcoord0);

                vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                for (const auto& primitive : mesh.primitives)
                {
                    pc.noise_texture_index = materials[primitive.material].basecolor_texture;
                    if (mi.variant_index == 1)
                    {
                        assert(pc.noise_texture_index >= 0);
                    }
                    vkCmdPushConstants(command_buffer, pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                    vkCmdDrawIndexed(command_buffer, primitive.index_count, 1, primitive.first_index, primitive.first_vertex, 0);
                }
            }

            vkCmdEndRendering(command_buffer);

            VkImageMemoryBarrier2 image_barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            image_barrier.srcStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
            image_barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            image_barrier.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            image_barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            image_barrier.image = depth_texture.image;
            image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            image_barrier.subresourceRange.baseArrayLayer = 0;
            image_barrier.subresourceRange.baseMipLevel = 0;
            image_barrier.subresourceRange.layerCount = 1;
            image_barrier.subresourceRange.levelCount = 1;

            VkDependencyInfo dependency_info{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency_info.imageMemoryBarrierCount = 1;
            dependency_info.pImageMemoryBarriers = &image_barrier;

            vkCmdPipelineBarrier2(command_buffer, &dependency_info);

            VkHelpers::end_label(command_buffer);
        }

        smoke_system.render(command_buffer, depth_texture);

        { // Forward pass
            VkHelpers::begin_label(command_buffer, "Forward pass", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

            VkRenderingAttachmentInfo color_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            color_info.imageView = hdr_render_target.view;
            color_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            color_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_info.clearValue.color = { 0.1f, 0.1f, 0.2f, 1.0f };

            VkRenderingAttachmentInfo depth_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            depth_info.imageView = depth_texture.view;
            depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth_info.storeOp = VK_ATTACHMENT_STORE_OP_NONE;
            depth_info.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            rendering_info.renderArea = { {0, 0}, {WINDOW_WIDTH, WINDOW_HEIGHT} };
            rendering_info.layerCount = 1;
            rendering_info.viewMask = 0;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_info;
            rendering_info.pDepthAttachment = &depth_info;

            vkCmdBeginRendering(command_buffer, &rendering_info);

            VkRect2D scissor = { {0, 0}, {WINDOW_WIDTH, WINDOW_HEIGHT} };
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            VkViewport viewport = { 0.0f, (float)WINDOW_HEIGHT, (float)WINDOW_WIDTH, -(float)WINDOW_HEIGHT, 0.0f, 1.0f };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline.layout, 1, 1, &ctx.bindless_descriptor_set, 0, nullptr);

            DescriptorInfo descriptor_info[] = {
                DescriptorInfo(anisotropic_sampler),
                DescriptorInfo(globals_buffer),
                DescriptorInfo(materials_buffer.buffer),
                DescriptorInfo(shadowmap_texture.view, VK_IMAGE_LAYOUT_GENERAL),
                DescriptorInfo(shadow_sampler),
                DescriptorInfo(point_sampler),
                DescriptorInfo(smoke_system.light_render_target.view, VK_IMAGE_LAYOUT_GENERAL),
            };

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, pipeline->pipeline.descriptor_update_template, pipeline->pipeline.layout, 0, descriptor_info);

            for (const auto& mi : mesh_draws)
            {
                const Mesh& mesh = meshes[mi.mesh_index];

                PushConstantsForward pc{};
                static_assert(sizeof(pc) <= 128);

                pc.model = mi.transform;
                pc.position_buffer = ctx.buffer_device_address(mesh.position);
                pc.disintegrate_alpha_reference = mi.variant_index != 0 ? disintegrate_alpha_reference : -100.0f;
                if (mesh.normal) pc.normal_buffer = ctx.buffer_device_address(mesh.normal);
                if (mesh.tangent) pc.tangent_buffer = ctx.buffer_device_address(mesh.tangent);
                if (mesh.texcoord0) pc.texcoord0_buffer = ctx.buffer_device_address(mesh.texcoord0);
                if (mesh.texcoord1) pc.texcoord1_buffer = ctx.buffer_device_address(mesh.texcoord1);

                vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                for (const auto& primitive : mesh.primitives)
                {
                    pc.material_index = primitive.material;

                    vkCmdPushConstants(command_buffer, pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                    vkCmdDrawIndexed(command_buffer, primitive.index_count, 1, primitive.first_index, primitive.first_vertex, 0);
                }
            }

            VkHelpers::end_label(command_buffer);
        }

        flow2.render(command_buffer);
        trail_blazer.render(command_buffer);
		particle_manager.render_systems(command_buffer);

        vkCmdEndRendering(command_buffer);

#if 0
        { // SDF test
            VkHelpers::begin_label(command_buffer, "SDF test", glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
            DescriptorInfo desc_info[] = {
                DescriptorInfo(globals_buffer),
                DescriptorInfo(bilinear_sampler),
                DescriptorInfo(sdf.texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                DescriptorInfo(hdr_render_target.view, VK_IMAGE_LAYOUT_GENERAL),
            };

            SDFPushConstants pc{};
            pc.grid_dims = sdf.dims;
            pc.grid_spacing = sdf.grid_spacing;
            pc.grid_origin = sdf.grid_origin;

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, sdf_test->pipeline.pipeline);
            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, sdf_test->pipeline.descriptor_update_template, sdf_test->pipeline.layout, 0, desc_info);
            vkCmdPushConstants(command_buffer, sdf_test->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(command_buffer, (WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

            VkHelpers::full_barrier(command_buffer);

            VkHelpers::end_label(command_buffer);
        }
#endif

        smoke_system.composite(command_buffer, hdr_render_target);

#if 0
        DescriptorInfo desc_info[] = {
            DescriptorInfo(gpu_particle_system.tlas.acceleration_structure),
            DescriptorInfo(globals_buffer.buffer),
            DescriptorInfo(hdr_render_target.view, VK_IMAGE_LAYOUT_GENERAL),
            DescriptorInfo(gpu_particle_system.particle_aabbs.buffer),
        };

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, test_pipeline->pipeline.pipeline);
        vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, test_pipeline->pipeline.descriptor_update_template, test_pipeline->pipeline.layout, 0, desc_info);
        vkCmdDispatch(command_buffer, (WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);
#endif

        {
            VkHelpers::begin_label(command_buffer, "Tonemap", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

            DescriptorInfo descriptor_info[] = {
                DescriptorInfo(hdr_render_target.view, VK_IMAGE_LAYOUT_GENERAL),
                DescriptorInfo(swapchain_texture.view, VK_IMAGE_LAYOUT_GENERAL)
            };

            PushConstantsTonemap pc{};
            pc.size = glm::uvec2(WINDOW_WIDTH, WINDOW_HEIGHT);

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, tonemap_pipeline->pipeline.descriptor_update_template, tonemap_pipeline->pipeline.layout, 0, descriptor_info);
            vkCmdPushConstants(command_buffer, tonemap_pipeline->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, tonemap_pipeline->pipeline.pipeline);

            uint32_t dispatch_x = get_golden_dispatch_size(WINDOW_WIDTH);
            uint32_t dispatch_y = get_golden_dispatch_size(WINDOW_HEIGHT);
            vkCmdDispatch(command_buffer, dispatch_x, dispatch_y, 1);

            VkHelpers::end_label(command_buffer);
        }

        {
            VkHelpers::begin_label(command_buffer, "ImGui render", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

            // ImGui rendering
            VkRenderingAttachmentInfo color_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            color_info.imageView = swapchain_texture.view;
            color_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            color_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingAttachmentInfo depth_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            depth_info.imageView = depth_texture.view;
            depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_info.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering_info{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            rendering_info.renderArea = { {0, 0}, {WINDOW_WIDTH, WINDOW_HEIGHT} };
            rendering_info.layerCount = 1;
            rendering_info.viewMask = 0;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_info;
            rendering_info.pDepthAttachment = &depth_info;

            vkCmdBeginRendering(command_buffer, &rendering_info);

            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();

            // Record dear imgui primitives into command buffer
            ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);

            vkCmdEndRendering(command_buffer);

            VkHelpers::end_label(command_buffer);
        }

        ctx.end_frame(command_buffer);
        frame_index++;

        timer.tock();
        cpu_time_ms = glm::mix(timer.get_elapsed_milliseconds(), cpu_time_ms, 0.95f);
    }

    hot_reload_watcher_should_quit = true;
    hot_reload_watcher.join();

    vkDeviceWaitIdle(ctx.device);
    vmaDestroyImage(ctx.allocator, shadowmap_texture.image, shadowmap_texture.allocation);
    vkDestroyImageView(ctx.device, shadowmap_texture.view, nullptr);
    vmaDestroyImage(ctx.allocator, depth_texture.image, depth_texture.allocation);
    vkDestroyImageView(ctx.device, depth_texture.view, nullptr);
    hdr_render_target.destroy(ctx.device, ctx.allocator);
    texture_catalog.shutdown();
    for (auto& m : meshes)
    {
        vmaDestroyBuffer(ctx.allocator, m.indices.buffer, m.indices.allocation);
        vmaDestroyBuffer(ctx.allocator, m.position.buffer, m.position.allocation);
        vmaDestroyBuffer(ctx.allocator, m.normal.buffer, m.normal.allocation);
        vmaDestroyBuffer(ctx.allocator, m.tangent.buffer, m.tangent.allocation);
        vmaDestroyBuffer(ctx.allocator, m.texcoord0.buffer, m.texcoord0.allocation);
        vmaDestroyBuffer(ctx.allocator, m.texcoord1.buffer, m.texcoord1.allocation);
    }
    ctx.destroy_buffer(materials_buffer);
    ctx.destroy_buffer(globals_buffer);
    ctx.destroy_buffer(mesh_disintegrate_spawn_positions);
    vkDestroySampler(ctx.device, anisotropic_sampler, nullptr);
    vkDestroySampler(ctx.device, bilinear_sampler, nullptr);
    vkDestroySampler(ctx.device, shadow_sampler, nullptr);
    vkDestroySampler(ctx.device, point_sampler, nullptr);
    for (auto& t : textures)
    {
        vkDestroyImageView(ctx.device, t.view, nullptr);
        vmaDestroyImage(ctx.allocator, t.image, t.allocation);
    }
    sdf.texture.destroy(ctx.device, ctx.allocator);
    smoke_system.destroy();
    flow2.destroy();
    trail_blazer.destroy();
    particle_manager.destroy();
    depth_prepass_disintegrate->builder.destroy_resources(depth_prepass_disintegrate->pipeline);
    depth_prepass->builder.destroy_resources(depth_prepass->pipeline);
    pipeline->builder.destroy_resources(pipeline->pipeline);
    shadowmap_pipeline->builder.destroy_resources(shadowmap_pipeline->pipeline);
    shadowmap_disintegrate_pipeline->builder.destroy_resources(shadowmap_disintegrate_pipeline->pipeline);
    procedural_skybox_pipeline->builder.destroy_resources(procedural_skybox_pipeline->pipeline);
    tonemap_pipeline->builder.destroy_resources(tonemap_pipeline->pipeline);
    test_pipeline->builder.destroy_resources(test_pipeline->pipeline);
    sdf_test->builder.destroy_resources(sdf_test->pipeline);
    particle_renderer.shutdown();

    ctx.shutdown();

    return 0;
}