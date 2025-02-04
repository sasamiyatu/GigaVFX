// Copyright 2021 The Fuchsia Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SRC_GRAPHICS_LIB_COMPUTE_RADIX_SORT_PLATFORMS_VK_TARGET_H_
#define SRC_GRAPHICS_LIB_COMPUTE_RADIX_SORT_PLATFORMS_VK_TARGET_H_

//
//
//

#include <stdint.h>

#include "target_requirements.h"

//
//
//

#define RS_HEADER_MAGIC 0x4B565352  // "RSVK"

//
// This structure packages target-specific configuration parameters.
//

struct radix_sort_vk_target_config
{
  uint32_t keyval_dwords;
  uint32_t disable_int64;

  struct
  {
    uint32_t workgroup_size_log2;
  } init;

  struct
  {
    uint32_t workgroup_size_log2;
    uint32_t block_rows;
  } fill;

  struct
  {
    uint32_t workgroup_size_log2;
    uint32_t subgroup_size_log2;
    uint32_t block_rows;
    uint32_t disable_smem_histogram;
  } histogram;

  struct
  {
    uint32_t workgroup_size_log2;
    uint32_t subgroup_size_log2;
  } prefix;

  struct
  {
    uint32_t workgroup_size_log2;
    uint32_t subgroup_size_log2;
    uint32_t block_rows;
    uint32_t enable_broadcast;
    uint32_t disable_reorder;
  } scatter;
};

//
// This structure packages target-specific SPIR-V modules.
//

struct radix_sort_vk_target_modules
{
  uint32_t         module_count;
  uint32_t         module_size[8];
  const uint32_t * module_data[8];
};

//
//
//

struct radix_sort_vk_target
{
  uint32_t                              magic;       // magic header dword
  union radix_sort_vk_target_extensions extensions;  // target device extensions
  union radix_sort_vk_target_features   features;    // target device features
  struct radix_sort_vk_target_config    config;      // target configuration
};

extern const struct radix_sort_vk_target_modules radix_sort_u32_modules_i64;
extern const struct radix_sort_vk_target_modules radix_sort_u64_modules_i64;
extern const struct radix_sort_vk_target_modules radix_sort_u32_modules_noi64;
extern const struct radix_sort_vk_target_modules radix_sort_u64_modules_noi64;

//
//
//

#endif  // SRC_GRAPHICS_LIB_COMPUTE_RADIX_SORT_PLATFORMS_VK_TARGET_H_
