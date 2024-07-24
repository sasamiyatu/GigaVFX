#pragma once

#define CGLTF_FLOAT_COUNT(accessor) (cgltf_num_components(accessor->type) * accessor->count)

struct Mesh;
struct cgltf_data;
struct Context;
struct Texture;
struct Material;

size_t load_meshes(Context& ctx, const cgltf_data* data, Mesh* out_meshes, size_t count);
size_t load_textures(Context& ctx, const cgltf_data* data, const char* gltf_path, Texture* out_textures, size_t count);
size_t load_materials(Context& ctx, const cgltf_data* data, Material* out_materials, size_t count);