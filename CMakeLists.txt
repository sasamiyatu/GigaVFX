cmake_minimum_required(VERSION 3.25)
project(GigaVFX VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(VULKAN_SDK_DIR $ENV{VULKAN_SDK})
set(SDL2_DIR ${VULKAN_SDK_DIR}/cmake)

find_package(Vulkan REQUIRED volk dxc)
find_package(SDL2 REQUIRED)

add_subdirectory(external/vk-radix-sort)

add_executable(gigavfx
    src/buffer.h
    src/camera.h
    src/cgltf.h
    src/defines.h
    src/gltf.h
    src/gltf.cpp
    src/gmath.h
    src/gmath.cpp
    src/gpu_particles.h
    src/gpu_particles.cpp
    src/graphics_context.h
    src/graphics_context.cpp
    src/hot_reload.h
    src/hot_reload.cpp
    src/main.cpp
    src/mesh.h
    src/misc.h
    src/log.h
    src/particle_system.h
    src/particle_system.cpp
    src/pipeline.h
    src/pipeline.cpp
    src/radix_sort.h
    src/radix_sort.cpp
    src/random.h
    src/shaders.h
    src/shaders.cpp
    src/spirv_reflect.c
    src/spirv_reflect.h
    src/stb_image.h
    src/texture.h
    src/texture.cpp
    src/texture_catalog.h
    src/texture_catalog.cpp
    src/VkBootstrap.h
    src/VkBootstrap.cpp
    src/VkBootstrapDispatch.h
    src/vk_helpers.h

    src/pcg/pcg_basic.h
    src/pcg/pcg_basic.cpp

    src/imgui/imconfig.h
    src/imgui/imgui_internal.h
    src/imgui/imstb_truetype.h
    src/imgui/imstb_textedit.h
    src/imgui/imstb_rectpack.h
    src/imgui/imgui.h
    src/imgui/imgui.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui_impl_vulkan.h
    src/imgui/imgui_impl_vulkan.cpp
    src/imgui/imgui_impl_sdl2.h
    src/imgui/imgui_impl_sdl2.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_demo.cpp
)

target_include_directories(gigavfx PUBLIC ${Vulkan_INCLUDE_DIRS})


target_link_libraries(gigavfx PUBLIC SDL2::SDL2 Vulkan::dxc_lib Vulkan::volk vk-radix-sort)
set_target_properties(
    gigavfx
    PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_compile_definitions(USE_PRECOMPILED_SHADERS)

if (WIN32)
    add_compile_definitions(NOMINMAX)
endif()