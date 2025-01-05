#pragma once 

#include <glm/glm.hpp>
#include <vector>

#include "texture.h"

struct Context;

// Represents a discretized SDF (i.e. 3D texture)
struct SDF
{
	glm::uvec3 dims = glm::uvec3(0);
	glm::vec3 grid_origin = glm::vec3(0.0f);
	float grid_spacing = 0.0f;
	std::vector<float> data;

	Texture texture = {};

	bool init_texture(Context& ctx);
};

// Loads .sdf file generated by SDFGen
// https://github.com/christopherbatty/SDFGen
bool sdf_load_from_file(SDF& out_sdf, const char* filepath);
