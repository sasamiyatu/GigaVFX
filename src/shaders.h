#pragma once
#include "defines.h"
#include <map>

struct ShaderSource
{
	std::string filepath;
	std::string entry_point;
	std::map<std::string, std::string> defines;

	ShaderSource() {};
	ShaderSource(const char* filepath, const char* entry_point)
		: filepath(filepath), entry_point(entry_point)
	{
	}

	void add_defines(const std::string& first, const std::string& second);
	void add_defines(const std::string& str);
};

namespace Shaders
{
	void init();
	uint32_t* load_shader(const char* filepath, const char* entry_point, VkShaderStageFlagBits shader_stage, uint32_t* size); // Deprecated
	uint32_t* load_shader(const ShaderSource& shader_source, VkShaderStageFlagBits shader_stage, uint32_t* size);
}