#include "gmath.h"
#include "defines.h"

float orient2d(const glm::vec2 a, const glm::vec2 b, const glm::vec2 c)
{
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

float point_inside_circle_2d(Circle circle, glm::vec2 point)
{
	return glm::distance(circle.center, point) < circle.radius;
}

Circle welzl_circle_2d(glm::vec2* points, int num_points, glm::vec2* support, int num_support)
{
	if (num_points == 0 || num_support == 3)
	{
		switch (num_support)
		{
		case 0:
			return Circle{ {}, 0.0f };
		case 1:
			return Circle{ support[0], 0.0f };
		case 2:
			return Circle{ (support[0] + support[1]) * 0.5f, glm::distance(support[0], support[1]) * 0.5f };
		case 3:
		{
			glm::vec2 sup[3];
			memcpy(sup, support, sizeof(sup));
			float det = orient2d(sup[0], sup[1], sup[2]);
			if (det < 0.0f) std::swap(sup[1], sup[2]); // Force counter-clockwise
			det = abs(det);
			float area = 0.5f * det;
			glm::vec2 V_10 = sup[1] - sup[0];
			glm::vec2 V_20 = sup[2] - sup[0];
			float L_10 = glm::dot(V_10, V_10);
			float L_20 = glm::dot(V_20, V_20);
			float x = sup[0].x + 1.0f / (4.0f * area) * ((sup[2].y - sup[0].y) * L_10 - (sup[1].y - sup[0].y) * L_20);
			float y = sup[0].y + 1.0f / (4.0f * area) * ((sup[1].x - sup[0].x) * L_20 - (sup[2].x - sup[0].x) * L_10);
			float dx = x - sup[0].x;
			float dy = y - sup[0].y;
			float r = sqrtf(dx * dx + dy * dy);
			return Circle{ glm::vec2(x, y), r };
		}
		default:
			assert(false);
			break;
		}
	}

	if (num_support > 3)
	{
		int br = 5;
	}
	int index = num_points - 1;
	Circle smallest_circle = welzl_circle_2d(points, num_points - 1, support, num_support);

	if (point_inside_circle_2d(smallest_circle, points[index]))
		return smallest_circle;

	support[num_support++] = points[index];
	return welzl_circle_2d(points, num_points - 1, support, num_support);
}

Sphere get_frustum_bounding_sphere(glm::mat4 projection)
{
	glm::vec3 frustum_points[4] = {
		glm::vec3(-1.0f, 1.0f, 1.0f),
		glm::vec3(1.0f, 1.0f, 1.0f),
		glm::vec3(-1.0f, -1.0f, 0.0f),
		glm::vec3(1.0f, -1.0f, 0.0f),
	};

	glm::vec3 frustum_points_view_space[4] = {};
	for (int i = 0; i < 4; ++i)
	{
		glm::vec4 unproject = glm::inverse(projection) * glm::vec4(frustum_points[i], 1.0f);
		frustum_points_view_space[i] = glm::vec3(unproject) / unproject.w;
	}

	// Project into 2D
	glm::vec3 x_axis = glm::normalize(frustum_points_view_space[3] - frustum_points_view_space[2]);
	glm::vec3 y_axis = glm::normalize(frustum_points_view_space[0] - frustum_points_view_space[2]);

	// Orthonormalization
	y_axis = glm::normalize(y_axis - glm::dot(x_axis, y_axis) * x_axis);
	assert(glm::abs(glm::dot(x_axis, y_axis)) < 1e-6);
	assert(glm::abs(glm::length(x_axis) - 1.0f) < 1e-6);
	assert(glm::abs(glm::length(y_axis) - 1.0f) < 1e-6);

	glm::vec2 points[4] = {};
	for (int i = 0; i < 4; ++i)
	{
		glm::vec2 p = glm::vec2(glm::dot(x_axis, frustum_points_view_space[i]), glm::dot(y_axis, frustum_points_view_space[i]));
		points[i] = p;
	}

	glm::vec2 sup[32];
	Circle circle = welzl_circle_2d(points, 4, sup, 0);
	glm::vec3 center = x_axis * circle.center.x + y_axis * circle.center.y;

	glm::vec3 dumb_center = glm::vec3(0.0f);
	for (int i = 0; i < 4; ++i)
	{
		dumb_center += frustum_points_view_space[i];
	}
	dumb_center /= 4.0f;

	float longest = 0.0f;
	for (int i = 0; i < 4; ++i)
	{
		float l = glm::distance(dumb_center, frustum_points_view_space[i]);
		if (l > longest) longest = l;
	}

	return Sphere{ center, circle.radius };
}
