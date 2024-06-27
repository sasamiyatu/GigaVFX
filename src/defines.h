#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#include <stdio.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>
#include "log.h"

#define VK_CHECK(x) \
    do {                                            \
        VkResult result = x;                        \
        if (result != VK_SUCCESS)                   \
        {                                           \
            LOG_ERROR("Vulkan error: %d", result);  \
            __debugbreak();                         \
        }                                           \
    } while(0)
