#pragma once
#include "defines.h"
#include <map>
#include <vector>
#include <filesystem>
#include <set>

struct ShaderSource
{
	struct SpecializationConstantEntry
	{
		enum Type
		{
			BOOL = 0,
			UINT,
			FLOAT,
		};

		Type type;
		uint32_t constant_id;
		union
		{
			VkBool32 bool_val;
			uint32_t uint_val;
			float float_val;
		};
	};

	std::string filepath;
	std::string entry_point;
	std::vector<std::string> prepend_lines;
	std::vector<std::string> append_lines;
	std::vector<SpecializationConstantEntry> specialization_constants;

	// Filled by shader compiler
	std::set<std::filesystem::path> dependencies;

	ShaderSource() {};
	ShaderSource(const char* filepath, const char* entry_point)
		: filepath(filepath), entry_point(entry_point ? entry_point : "")
	{
	}

	void add_defines(const std::string& first, const std::string& second);
	void add_defines(const std::string& str);
	void add_include(const std::string& str, bool append = false);
	void add_specialization_constant(uint32_t constant_id, bool value);
	void add_specialization_constant(uint32_t constant_id, uint32_t value);
	void add_specialization_constant(uint32_t constant_id, float value);
};

namespace Shaders
{
	void init();
	uint32_t* load_shader(const char* filepath, const char* entry_point, VkShaderStageFlagBits shader_stage, uint32_t* size); // Deprecated
	uint32_t* load_shader(ShaderSource& shader_source, VkShaderStageFlagBits shader_stage, uint32_t* size);
}