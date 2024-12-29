#pragma once
#include <glm/glm.hpp>

struct CameraState
{
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    float fov = glm::radians(60.0f);
    float znear = 0.1f;
    float zfar = 100.0f;
};