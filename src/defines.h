#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include <Volk/volk.h>
#include <stdio.h>
#include <vulkan/vk_enum_string_helper.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>
#include "log.h"

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/euler_angles.hpp"


#define VK_CHECK(x) \
    do {                                            \
        VkResult result = x;                        \
        if (result != VK_SUCCESS)                   \
        {                                           \
            LOG_ERROR("Vulkan error: %d", result);  \
            __debugbreak();                         \
        }                                           \
    } while(0)
