cmake_minimum_required(VERSION 3.25)
project(Gigasticle VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(VULKAN_SDK_DIR $ENV{VULKAN_SDK})
set(SDL2_DIR ${VULKAN_SDK_DIR}/cmake)

find_package(Vulkan REQUIRED volk dxc)
find_package(SDL2 REQUIRED)

#===========================================================================================
# WGSL Shader precompilation
#
# Shaders are first from HLSL to Spir-V using DirectX Shader Compiler (DXC)
# Generated Spir-V is then transpiled into WGSL using Google's Tint shader compiler
# The generated WGSL shaders are embedded into the program using CMakeRC
# (https://github.com/vector-of-bool/cmrc)
#==========================================================================================

OPTION(WEBGPU_USE_PRECOMPILED_SHADERS "Use precompiled shaders when building for WebGPU" OFF)

if(WEBGPU_USE_PRECOMPILED_SHADERS)
    add_compile_definitions(USE_PRECOMPILED_SHADERS)
endif()

find_program(DXC dxc REQUIRED)
include(CMakeRC.cmake)

set(COMPILED_SHADERS)
function(compile_shader src_file type entry_point)
    get_filename_component(OUTPUT_FILE_WE ${src_file} NAME_WE)
    set(OUTPUT_SPV_FILE ${CMAKE_SOURCE_DIR}/shaders/generated/${OUTPUT_FILE_WE}_${type}.spv)
    set(SOURCE_FILE ${CMAKE_SOURCE_DIR}/${src_file})
    add_custom_command(
        OUTPUT ${OUTPUT_SPV_FILE}
        COMMAND ${DXC} ARGS -E ${entry_point} -T ${type} -Zs -spirv -O0 -Fo ${OUTPUT_SPV_FILE} ${SOURCE_FILE}
        DEPENDS ${SOURCE_FILE}
        VERBATIM
    )
    set(COMPILED_SHADERS ${COMPILED_SHADERS} ${OUTPUT_SPV_FILE} PARENT_SCOPE)
    list(APPEND SHADER_SOURCES ${SOURCE_FILE})
endfunction()

compile_shader(shaders/triangle.hlsl vs_6_6 vs_main)
compile_shader(shaders/triangle.hlsl ps_6_6 fs_main)
compile_shader(shaders/forward.hlsl vs_6_6 vs_main)
compile_shader(shaders/forward.hlsl ps_6_6 fs_main)
compile_shader(shaders/procedural_sky.hlsl cs_6_6 cs_main)

cmrc_add_resource_library(embedded_shaders ALIAS Shaders::precompiled
    WHENCE shaders/generated
    PREFIX shaders/
    ${COMPILED_SHADERS}
)

add_executable(gigasticle
    src/buffer.h
    src/cgltf.h
    src/defines.h
    src/gltf.h
    src/gltf.cpp
    src/gmath.h
    src/gmath.cpp
    src/graphics_context.h
    src/graphics_context.cpp
    src/hot_reload.h
    src/hot_reload.cpp
    src/main.cpp
    src/mesh.h
    src/misc.h
    src/log.h
    src/pipeline.h
    src/pipeline.cpp
    src/shaders.h
    src/shaders.cpp
    src/spirv_reflect.c
    src/spirv_reflect.h
    src/spirv_reflect.cpp
    src/stb_image.h
    src/texture.h
    src/VkBootstrap.h
    src/VkBootstrap.cpp
    src/VkBootstrapDispatch.h
    src/vk_helpers.h

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

target_include_directories(gigasticle PUBLIC ${Vulkan_INCLUDE_DIRS})

add_custom_target(shader_compilation ALL DEPENDS ${COMPILED_SHADERS})
target_link_libraries(gigasticle PUBLIC Shaders::precompiled)

#===========================================================================================
# End shader stuff
#===========================================================================================

target_link_libraries(gigasticle PUBLIC SDL2::SDL2 Vulkan::Vulkan Vulkan::dxc_lib)
set_target_properties(
    gigasticle
    PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_compile_definitions(USE_PRECOMPILED_SHADERS)

if (WIN32)
    add_compile_definitions(NOMINMAX)
endif()