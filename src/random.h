#pragma once

#include "pcg/pcg_basic.h"
#include <math.h>
#include "gmath.h"

inline float uniform_random()
{
	const float scale = ldexpf(1.0f, -32);
	return pcg32_random() * scale;
}

inline float random_in_range(float low, float high)
{
	if (high < low) std::swap(low, high);
	float range = high - low;
	return uniform_random() * range + low;
}

inline int random_int_in_range(int low, int high)
{
	assert(high >= low);
	int range = high - low;
	return pcg32_random() % range + low;
}

// Returns a random vector within a cone oriented towards the +z axis
inline glm::vec3 random_vector_in_cone(float min_angle_cos)
{
	assert(min_angle_cos >= -1.0f);
	assert(min_angle_cos <= 1.0f);
	const float z = random_in_range(min_angle_cos, 1.0f);
	const float phi = random_in_range(0.0f, 2.0f * M_PI);
	const float s = sqrtf(1.0f - z * z);
	return glm::vec3(s * cosf(phi), s * sinf(phi), z);
}

inline glm::vec3 random_vector_in_oriented_cone(float min_angle_cos, glm::vec3 cone_dir)
{
	assert(near_one(cone_dir));
	glm::vec3 v = random_vector_in_cone(min_angle_cos);
	
	const glm::vec3 z_axis = glm::vec3(0.0f, 0.0f, 1.0f);
	float cos_theta = glm::dot(cone_dir, z_axis);
	if (fabsf(cos_theta) > 0.99f)
	{
		return glm::sign(cos_theta) * v;
	}
	else
	{
		glm::vec3 rotation_axis = glm::cross(z_axis, cone_dir);
		float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
		glm::vec3 rotated = cos_theta * v + sin_theta * glm::cross(rotation_axis, v) + (1.0f - cos_theta) * glm::dot(rotation_axis, v) * rotation_axis;
		return rotated;
	}
}

template <typename T>
inline T random_vector()
{
	T type;
	constexpr size_t len = type.length();
	for (size_t i = 0; i < len; ++i)
	{
		type[i] = uniform_random();
	}

	return type;
}