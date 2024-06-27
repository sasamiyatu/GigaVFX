#define VK_USE_PLATFORM_WIN32_KHR
#include <stdio.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>
#include "log.h"
#include "VkBootstrap.h"

#define VK_CHECK(x) \
    do {                                            \
        VkResult result = x;                        \
        if (result != VK_SUCCESS)                   \
        {                                           \
            LOG_ERROR("Vulkan error: %d", result);  \
            __debugbreak();                         \
        }                                           \
    } while(0)


constexpr uint32_t window_width = 1280;
constexpr uint32_t window_height = 720;

struct Context
{
    SDL_Window* window;
    vkb::Instance instance;
    VkSurfaceKHR surface;
    vkb::PhysicalDevice physical_device;
    vkb::Device device;
    VkQueue graphics_queue;
    vkb::Swapchain swapchain;
};

Context ctx;

int main()
{
    SDL_Init(SDL_INIT_VIDEO);

    ctx.window = SDL_CreateWindow("Gigasticle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_VULKAN);

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
        return -1;
    }

    ctx.instance = instance_result.value();

    auto system_info_ret = vkb::SystemInfo::get_system_info();
    if (!system_info_ret) {
        LOG_ERROR("%s\n", system_info_ret.error().message());
        return -1;
    }
    auto system_info = system_info_ret.value();

    if (!SDL_Vulkan_CreateSurface(ctx.window, ctx.instance, &ctx.surface))
    {
        LOG_ERROR("Failed to create Vulkan surface");
        exit(-1);
    }

    vkb::PhysicalDeviceSelector phys_device_selector(ctx.instance);
    auto physical_device_selector_return = phys_device_selector
        .set_surface(ctx.surface)
        .select();
    if (!physical_device_selector_return) {
        // Handle error
        LOG_ERROR("Failed to create Vulkan physical device!");
        exit(-1);
    }
    ctx.physical_device = physical_device_selector_return.value();
    LOG_INFO("Selected physical device: %s", ctx.physical_device.name.c_str());
    std::vector<std::string> available_device_exts = ctx.physical_device.get_available_extensions();
    LOG_INFO("Available device extensions:");
    for (const auto& e : available_device_exts)
    {
        LOG_INFO("%s", e.c_str());
    }

    vkb::DeviceBuilder device_builder{ ctx.physical_device };
    auto dev_ret = device_builder.build();
    if (!dev_ret) {
        // error
        LOG_ERROR("Failed to create Vulkan device!");
        exit(-1);
    }
    ctx.device = dev_ret.value();
    ctx.graphics_queue = ctx.device.get_queue(vkb::QueueType::graphics).value();

    vkb::SwapchainBuilder swapchain_builder{ ctx.device };
    auto swap_ret = swapchain_builder.build();
    if (!swap_ret) {
        LOG_ERROR("Failed to create swapchain!");
        exit(-1);
    }
    ctx.swapchain = swap_ret.value();
    
    LOG_DEBUG("Swapchain format: %s", string_VkFormat(ctx.swapchain.image_format));

    bool running = true;

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
            default:
                break;
            }
        }
    }

    vkb::destroy_swapchain(ctx.swapchain);
    vkb::destroy_device(ctx.device);
    vkDestroySurfaceKHR(ctx.instance, ctx.surface, nullptr);
    vkb::destroy_instance(ctx.instance);

    SDL_DestroyWindow(ctx.window);
    SDL_Quit();

    return 0;
}