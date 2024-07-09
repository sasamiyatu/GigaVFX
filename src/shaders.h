#pragma once
#include "defines.h"

namespace Shaders
{
	void init();
	uint32_t* load_shader(const char* filepath, VkShaderStageFlagBits shader_stage, uint32_t* size);
}