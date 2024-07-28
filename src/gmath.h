#pragma once
#include "defines.h"

struct Circle
{
	glm::vec2 center;
	float radius;
};

struct Sphere
{
	glm::vec3 center;
	float radius;
};

float orient2d(const glm::vec2 a, const glm::vec2 b, const glm::vec2 c);
float point_inside_circle_2d(Circle circle, glm::vec2 point);

Circle welzl_circle_2d(glm::vec2* points, int num_points, glm::vec2* support, int num_support);


Sphere get_frustum_bounding_sphere(glm::mat4 projection);

inline bool near_one(glm::vec3 v)
{
	constexpr float threshold = 1e-5f;
	float mag2 = glm::dot(v, v);
	return fabsf(mag2 - 1.0f) < threshold;
}