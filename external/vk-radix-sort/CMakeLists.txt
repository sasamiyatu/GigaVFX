cmake_minimum_required(VERSION 3.16)

project(RadixSort LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED volk)

function(shader header source source_list)
    get_filename_component(shader_bin_name ${header} NAME_WLE)
    string(REPLACE "\." "_" shader_bin_name ${shader_bin_name})
    add_custom_command(
        OUTPUT ${header}
        COMMAND glslangValidator --target-env vulkan1.2 --vn ${shader_bin_name}_shader_binary ${CMAKE_CURRENT_SOURCE_DIR}/${source} -o ${CMAKE_CURRENT_BINARY_DIR}/${header} ${ARGN}
        MAIN_DEPENDENCY ${source}
        DEPENDS ${source}
        IMPLICIT_DEPENDS C ${source}
        VERBATIM
    )
    set(${source_list} ${${source_list}} ${header} PARENT_SCOPE)
endfunction()

foreach(support IN ITEMS noi64 i64)
    foreach(keyval IN ITEMS u32 u64)
        set(a ${support}_${keyval}_)
        set(d )
        if(${support} STREQUAL noi64)
          set(d ${d} -DRS_DISABLE_SHADER_INT64)
        endif()
        if(${keyval} STREQUAL u32)
          set(d ${d} -DRS_KEYVAL_DWORDS=1)
        else()
          set(d ${d} -DRS_KEYVAL_DWORDS=2)
        endif()
        shader(${a}init.comp.h lib/shaders/init.comp shader_binary ${d})
        shader(${a}fill.comp.h lib/shaders/fill.comp shader_binary ${d})
        shader(${a}histogram.comp.h lib/shaders/histogram.comp shader_binary ${d})
        shader(${a}prefix.comp.h lib/shaders/prefix.comp shader_binary ${d})
        shader(${a}scatter.comp.h lib/shaders/scatter.comp shader_binary ${d})
    endforeach()
endforeach()

set(targets
    amd/gcn3
    arm/bifrost4
    arm/bifrost8
    intel/gen8
    nvidia/sm35
)

foreach(target ${targets})
    set(target_sources ${target_sources} "${CMAKE_CURRENT_SOURCE_DIR}/lib/targets/${target}/u32/header.c")
    set(target_sources ${target_sources} "${CMAKE_CURRENT_SOURCE_DIR}/lib/targets/${target}/u64/header.c")
endforeach()

add_library(vk-radix-sort STATIC
    lib/radix_sort_vk.c
    lib/radix_sort_vk_ext.h
    lib/shaders/push.h
    lib/target.h
    lib/target.c
    lib/target_requirements.c
    lib/target_requirements.h
    common/vk/assert.c
    common/vk/barrier.c
    common/util.c
    ${target_sources}
    ${shader_binary}
)

#target_link_libraries(vk-radix-sort PUBLIC Vulkan::Vulkan)
target_link_libraries(vk-radix-sort PUBLIC Vulkan::volk)
target_include_directories(vk-radix-sort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(vk-radix-sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        enable_language(CXX)
        set(common_src
            tests/radix_sort_vk_bench/cpu_sort.cc
            tests/radix_sort_vk_bench/radix_sort_vk_bench.c
            tests/radix_sort_vk_bench/find_target_name.c
            common/vk/assert.c
            common/vk/barrier.c
            common/vk/debug_utils.c
            common/vk/find_mem_type_idx.c
            common/vk/pipeline_cache.c
            common/util.c
        )
        add_executable(benchmark tests/radix_sort_vk_bench/main.c ${common_src})
        target_link_libraries(benchmark PUBLIC vk-radix-sort)
        target_include_directories(benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(benchmark PUBLIC RS_VK_TARGET_ARCHIVE_LINKABLE)

        include(GoogleTest)

        add_executable(unittest tests/radix_sort_vk_bench/tests/radix_sort_vk_bench_unittests.cc ${common_src})
        target_link_libraries(unittest PUBLIC vk-radix-sort gtest gtest_main)
        target_include_directories(unittest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} tests/radix_sort_vk_bench)
        target_compile_definitions(unittest PUBLIC RS_VK_TARGET_ARCHIVE_LINKABLE)

        gtest_add_tests(unittest "" AUTO)
    endif()
endif()

