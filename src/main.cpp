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
#include "gmath.h"
#include "hot_reload.h"
#include "mesh.h"
#include "buffer.h"
#include "texture.h"
#include "gltf.h"
#include "graphics_context.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl2.h"
#include "imgui/imgui_impl_vulkan.h"

constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;

constexpr uint32_t MAX_BINDLESS_RESOURCES = 1024;

struct CameraState
{
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    float fov = glm::radians(75.0f);
    float znear = 0.1f;
    float zfar = 100.0f;
};

Context ctx;

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
};

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

    ctx.init(WINDOW_WIDTH, WINDOW_HEIGHT);

    init_imgui();

    // Init shader compiler
    Shaders::init();

    VkSampler sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.maxLod = VK_LOD_CLAMP_NONE;
        info.anisotropyEnable = VK_TRUE;
        info.maxAnisotropy = ctx.physical_device.properties.limits.maxSamplerAnisotropy;
        VK_CHECK(vkCreateSampler(ctx.device, &info, nullptr, &sampler));
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
    VkSampler point_sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        info.magFilter = VK_FILTER_NEAREST;
        info.minFilter = VK_FILTER_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VK_CHECK(vkCreateSampler(ctx.device, &info, nullptr, &point_sampler));
    }

    Texture depth_texture;
    ctx.create_texture(depth_texture, WINDOW_WIDTH, WINDOW_HEIGHT, 1u, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

    Texture shadowmap_texture;
    ctx.create_texture(shadowmap_texture, 2048, 2048, 1u, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 4);

    GraphicsPipelineBuilder pipeline_builder(ctx.device, true);
    pipeline_builder
        .set_vertex_shader_filepath("forward.hlsl")
        .set_fragment_shader_filepath("forward.hlsl")
        .add_color_attachment(ctx.swapchain.image_format)
        .set_cull_mode(VK_CULL_MODE_NONE)
        .set_depth_format(VK_FORMAT_D32_SFLOAT)
        .set_depth_test(VK_TRUE)
        .set_depth_write(VK_TRUE)
        .set_depth_compare_op(VK_COMPARE_OP_LESS)
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

    ComputePipelineBuilder compute_builder(ctx.device, true);
    compute_builder
        .set_shader_filepath("procedural_sky.hlsl");
    ComputePipelineAsset* procedural_skybox_pipeline = new ComputePipelineAsset(compute_builder);
    AssetCatalog::register_asset(procedural_skybox_pipeline);

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
    desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    Buffer globals_buffer = ctx.create_buffer(desc);

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

    std::vector<MeshInstance> mesh_draws;

    bool running = true;
    while (running)
    {
        VkCommandBuffer command_buffer = ctx.begin_frame();
        Texture& swapchain_texture = ctx.get_swapchain_texture();

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

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

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {

            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

            ImGui::End();
        }

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

        static float hot_reload_timer = 0.0f;
        hot_reload_timer += delta_time;
        if (hot_reload_timer > 1.0f)
        {
            if (AssetCatalog::check_for_dirty_assets())
            {
                VK_CHECK(vkDeviceWaitIdle(ctx.device));
                while (!AssetCatalog::reload_dirty_assets())
                {
                    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Shader compilation error", "Shader compilation failed!\nRetry?", ctx.window);
                }
            }
            hot_reload_timer -= 1.0f;
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
                        cgltf_node_transform_world(node, glm::value_ptr(mi.transform));
                    }
                };

            const cgltf_scene* scene = gltf_data->scene;
            for (size_t i = 0; i < scene->nodes_count; ++i)
            {
                traverse_tree(scene->nodes[i], get_meshes);
            }
        }

        { // Update global uniform buffer
            ShaderGlobals globals{};
            globals.view = glm::lookAt(camera.position, camera.position + camera.forward, camera.up);
            globals.view_inverse = glm::inverse(globals.view);
            globals.projection = glm::perspective(camera.fov, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, camera.znear, camera.zfar);
            globals.viewprojection = globals.projection * globals.view;
            globals.camera_pos = glm::vec4(camera.position, 1.0f);
            glm::vec3 sundir = glm::normalize(glm::vec3(1.0f));
            globals.sun_direction = glm::vec4(sundir, 1.0f);
            globals.sun_color_and_intensity = glm::vec4(1.0f);
            //globals.shadow_view = glm::lookAt(camera.position + sundir * 30.0f, camera.position, glm::vec3(0.0f, 1.0f, 0.0f));
            //globals.shadow_projection = glm::ortho(-30.0f, 30.0f, -30.0f, 30.0f, 0.1f, 100.0f);
            //globals.shadow_view_projection = globals.shadow_projection * globals.shadow_view;
            //float znear = globals.shadow_projection[3][2] / globals.shadow_projection[2][2];
            //float zfar_minus_znear = -1.0f / globals.shadow_projection[2][2];
            //globals.shadow_projection_info = glm::vec4(zfar_minus_znear, znear, 0.0f, 0.0f);


            glm::vec4 origin_shift[4];
            float max_distance = 100.0f;
            glm::mat4 shadow_projs[4];
            glm::mat4 shadow_views[4];
            glm::mat4 shadow_view_projs[4];
            float distance_thresholds[4] = { 0.0f, 5.0, 15.0f, 45.0f };
            memcpy(glm::value_ptr(globals.shadow_cascade_thresholds), distance_thresholds, sizeof(distance_thresholds));
            LOG_DEBUG("Pos: %f %f %f", camera.position.x, camera.position.y, camera.position.z);
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

                glm::vec4 shadow_origin = shadow_view_projs[i] * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                shadow_origin /= shadow_origin.w;
                float step = 1.0f / 1024.0f;
                glm::vec2 rounded = glm::round(glm::vec2(shadow_origin) / step) * step;
                globals.shadow_view_projection[i] = glm::translate(glm::mat4(1.0f), glm::vec3(rounded.x - shadow_origin.x, rounded.y - shadow_origin.y, 0.0f)) * globals.shadow_view_projection[i];
            }

            int index = 0;
            //globals.shadow_view = shadow_views[index];
            //globals.shadow_projection = shadow_projs[index];
            //globals.shadow_view_projection = shadow_view_projs[index];
            //float znear = globals.shadow_projection[3][2] / globals.shadow_projection[2][2];
            //float zfar_minus_znear = -1.0f / globals.shadow_projection[2][2];
            //globals.shadow_projection_info = glm::vec4(zfar_minus_znear, znear, 0.0f, 0.0f);


            void* mapped;
            vmaMapMemory(ctx.allocator, globals_buffer.allocation, &mapped);
            memcpy(mapped, &globals, sizeof(globals));
            vmaUnmapMemory(ctx.allocator, globals_buffer.allocation);
        }

        { // Transition depth buffer layout
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

        { // Procedural sky box
            DescriptorInfo descriptor_info[] = {
                DescriptorInfo(globals_buffer.buffer),
                DescriptorInfo(swapchain_texture.view, VK_IMAGE_LAYOUT_GENERAL)
            };

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, procedural_skybox_pipeline->pipeline.descriptor_update_template, procedural_skybox_pipeline->pipeline.layout, 0, descriptor_info);
            
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, procedural_skybox_pipeline->pipeline.pipeline);

            uint32_t dispatch_x = get_golden_dispatch_size(WINDOW_WIDTH);
            uint32_t dispatch_y = get_golden_dispatch_size(WINDOW_HEIGHT);
            vkCmdDispatch(command_buffer, dispatch_x, dispatch_y, 1);
        }

        { // Shadowmap

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
            rendering_info.renderArea = { {0, 0}, {2048, 2048} };
            rendering_info.layerCount = 4;
            rendering_info.viewMask = 0b1111;
            rendering_info.pDepthAttachment = &depth_info;

            VkRect2D scissor = { {0, 0}, {2048, 2048} };
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            VkViewport viewport = { 0.0f, (float)2048, (float)2048, -(float)2048, 0.0f, 1.0f };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            vkCmdBeginRendering(command_buffer, &rendering_info);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowmap_pipeline->pipeline.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowmap_pipeline->pipeline.layout, 1, 1, &ctx.bindless_descriptor_set, 0, nullptr);

            DescriptorInfo descriptor_info[] = {
                DescriptorInfo(sampler),
                DescriptorInfo(globals_buffer.buffer),
                DescriptorInfo(materials_buffer.buffer)
            };

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, shadowmap_pipeline->pipeline.descriptor_update_template, shadowmap_pipeline->pipeline.layout, 0, descriptor_info);

            for (const auto& mi : mesh_draws)
            {
                const Mesh& mesh = meshes[mi.mesh_index];

                PushConstantsForward pc{};
                static_assert(sizeof(pc) <= 128);

                pc.model = mi.transform;

                pc.position_buffer = ctx.buffer_device_address(mesh.position);
                if (mesh.normal) pc.normal_buffer = ctx.buffer_device_address(mesh.normal);
                if (mesh.tangent) pc.tangent_buffer = ctx.buffer_device_address(mesh.tangent);
                if (mesh.texcoord0) pc.texcoord0_buffer = ctx.buffer_device_address(mesh.texcoord0);
                if (mesh.texcoord1) pc.texcoord1_buffer = ctx.buffer_device_address(mesh.texcoord1);

                vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                for (const auto& primitive : mesh.primitives)
                {
                    pc.material_index = primitive.material;

                    vkCmdPushConstants(command_buffer, shadowmap_pipeline->pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                    vkCmdDrawIndexed(command_buffer, primitive.index_count, 1, primitive.first_index, primitive.first_vertex, 0);
                }
            }

            vkCmdEndRendering(command_buffer);
        }

        { // Forward pass
            VkRenderingAttachmentInfo color_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            color_info.imageView = swapchain_texture.view;
            color_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            color_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_info.clearValue.color = { 0.1f, 0.1f, 0.2f, 1.0f };

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
                DescriptorInfo(sampler),
                DescriptorInfo(globals_buffer.buffer),
                DescriptorInfo(materials_buffer.buffer),
                DescriptorInfo(shadowmap_texture.view, VK_IMAGE_LAYOUT_GENERAL),
                DescriptorInfo(shadow_sampler),
                DescriptorInfo(point_sampler),
            };

            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, pipeline->pipeline.descriptor_update_template, pipeline->pipeline.layout, 0, descriptor_info);

            for (const auto& mi : mesh_draws)
            {
                const Mesh& mesh = meshes[mi.mesh_index];

                PushConstantsForward pc{};
                static_assert(sizeof(pc) <= 128);

                pc.model = mi.transform;
                pc.position_buffer = ctx.buffer_device_address(mesh.position);
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
        }

        {
            // ImGui rendering
            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();

            // Record dear imgui primitives into command buffer
            ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
        }

        vkCmdEndRendering(command_buffer);

        ctx.end_frame(command_buffer);
    }

    vkDeviceWaitIdle(ctx.device);
    vmaDestroyImage(ctx.allocator, shadowmap_texture.image, shadowmap_texture.allocation);
    vkDestroyImageView(ctx.device, shadowmap_texture.view, nullptr);
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
    vmaDestroyBuffer(ctx.allocator, materials_buffer.buffer, materials_buffer.allocation);
    vmaDestroyBuffer(ctx.allocator, globals_buffer.buffer, globals_buffer.allocation);
    vkDestroySampler(ctx.device, sampler, nullptr);
    vkDestroySampler(ctx.device, shadow_sampler, nullptr);
    vkDestroySampler(ctx.device, point_sampler, nullptr);
    for (auto& t : textures)
    {
        vkDestroyImageView(ctx.device, t.view, nullptr);
        vmaDestroyImage(ctx.allocator, t.image, t.allocation);
    }
    pipeline->builder.destroy_resources(pipeline->pipeline);
    shadowmap_pipeline->builder.destroy_resources(shadowmap_pipeline->pipeline);
    procedural_skybox_pipeline->builder.destroy_resources(procedural_skybox_pipeline->pipeline);

    ctx.shutdown();

    return 0;
}