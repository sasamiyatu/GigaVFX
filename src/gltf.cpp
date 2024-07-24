#include "gltf.h"
#include "cgltf.h"
#include "mesh.h"
#include "graphics_context.h"
#include "stb_image.h"
#include "../shaders/shared.h"

#include <vector>

size_t load_meshes(Context& ctx, const cgltf_data* gltf_data, Mesh* out_meshes, size_t count)
{
    if (out_meshes == nullptr) return gltf_data->meshes_count;

    for (size_t i = 0; i < count; ++i)
    {
        const cgltf_mesh& m = gltf_data->meshes[i];
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> normal;
        std::vector<glm::vec4> tangent;
        std::vector<glm::vec2> texcoord0;
        std::vector<glm::vec2> texcoord1;
        std::vector<uint32_t> indices;

        Mesh& mesh = out_meshes[i];
        mesh.primitives.resize(m.primitives_count);

        for (size_t j = 0; j < m.primitives_count; ++j)
        {
            const cgltf_primitive& p = m.primitives[j];
            assert(p.indices);

            Mesh::Primitive& primitive = mesh.primitives[j];

            uint32_t first_vertex = position.size();
            uint32_t first_index = indices.size();
            uint32_t index_count = p.indices->count;

            primitive.first_vertex = first_vertex;
            primitive.first_index = first_index;
            primitive.index_count = index_count;
            primitive.material = cgltf_material_index(gltf_data, p.material);

            indices.resize(first_index + index_count);
            cgltf_accessor_unpack_indices(p.indices, indices.data() + first_index, sizeof(uint32_t), index_count);

            bool primitive_has_tangents = false;
            for (size_t k = 0; k < p.attributes_count; ++k)
            {
                const cgltf_attribute& a = p.attributes[k];
                switch (a.type)
                {
                case cgltf_attribute_type_position:
                {
                    assert(a.data->component_type == cgltf_component_type_r_32f);
                    assert(a.data->type == cgltf_type_vec3); // TODO: Handle other component types and types
                    position.resize(first_vertex + a.data->count);
                    cgltf_accessor_unpack_floats(a.data, (cgltf_float*)(position.data() + first_vertex), CGLTF_FLOAT_COUNT(a.data));
                    break;
                }
                case cgltf_attribute_type_normal:
                {
                    assert(a.data->component_type == cgltf_component_type_r_32f);
                    assert(a.data->type == cgltf_type_vec3); // TODO: Handle other component types and types
                    normal.resize(first_vertex + a.data->count);
                    cgltf_accessor_unpack_floats(a.data, (cgltf_float*)(normal.data() + first_vertex), CGLTF_FLOAT_COUNT(a.data));
                    break;
                }
                case cgltf_attribute_type_tangent:
                {
                    primitive_has_tangents = true;
                    assert(a.data->component_type == cgltf_component_type_r_32f);
                    assert(a.data->type == cgltf_type_vec4); // TODO: Handle other component types and types
                    tangent.resize(first_vertex + a.data->count);
                    cgltf_accessor_unpack_floats(a.data, (cgltf_float*)(tangent.data() + first_vertex), CGLTF_FLOAT_COUNT(a.data));
                    break;
                }
                case cgltf_attribute_type_texcoord:
                {
                    cgltf_float* out = nullptr;
                    switch (a.index)
                    {
                    case 0:
                        texcoord0.resize(first_vertex + a.data->count);
                        out = (cgltf_float*)(texcoord0.data() + first_vertex);
                        break;
                    case 1:
                        texcoord1.resize(first_vertex + a.data->count);
                        out = (cgltf_float*)(texcoord1.data() + first_vertex);
                        break;
                    default:
                        LOG_WARNING("Unused texcoord index: %d", a.index);
                        break;
                    }
                    cgltf_accessor_unpack_floats(a.data, out, CGLTF_FLOAT_COUNT(a.data));

                    break;
                }
                default:
                    LOG_WARNING("Unused gltf attribute: %s", a.name);
                    break;
                }
            }

            if (p.material->normal_texture.texture && !primitive_has_tangents)
            {
                LOG_WARNING("Primitive on mesh %s has a normal map but is missing tangents!", m.name ? m.name : "");
            }
        }

        // Create GPU buffers
        { // Indices
            assert(!indices.empty());
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(indices);
            desc.usage_flags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = indices.data();
            mesh.indices = ctx.create_buffer(desc);
        }
        { // Position
            assert(!position.empty());
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(position);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = position.data();
            mesh.position = ctx.create_buffer(desc);
        }
        if (!normal.empty())
        { // Normal
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(normal);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = normal.data();
            mesh.normal = ctx.create_buffer(desc);
        }
        else
        {
            LOG_WARNING("Mesh has no normals!");
        }
        if (!tangent.empty())
        { // Tangent
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(tangent);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = tangent.data();
            mesh.tangent = ctx.create_buffer(desc);
        }
        else
        {
            LOG_WARNING("Mesh has no tangents!");
        }
        if (!texcoord0.empty())
        { // Texcoord0
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(texcoord0);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = texcoord0.data();
            mesh.texcoord0 = ctx.create_buffer(desc);
        }
        if (!texcoord1.empty())
        { // Texcoord1
            BufferDesc desc{};
            desc.size = VECTOR_SIZE_BYTES(texcoord1);
            desc.usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            desc.data = texcoord1.data();
            mesh.texcoord1 = ctx.create_buffer(desc);
        }
    }

    return count;
}

size_t load_textures(Context& ctx, const cgltf_data* gltf_data, const char* gltf_path, Texture* out_textures, size_t count)
{
    if (out_textures == nullptr) return gltf_data->textures_count;

    for (size_t i = 0; i < count; ++i)
    {
        const cgltf_texture& tex = gltf_data->textures[i];

        Texture& t = out_textures[i];

        if (tex.image->buffer_view)
        {
            int comp;
            t.source = stbi_load_from_memory((stbi_uc*)cgltf_buffer_view_data(tex.image->buffer_view), tex.image->buffer_view->size, &t.width, &t.height, &comp, 4);
            assert(t.source);
        }
        else
        {
            std::string path = gltf_path;
            std::string uri = tex.image->uri;
            size_t last_slash = path.find_last_of('/');
            path = path.substr(0, last_slash + 1) + uri;
            int comp;
            t.source = stbi_load(path.c_str(), &t.width, &t.height, &comp, 4);
            assert(t.source);
        }
    }

    if (!ctx.create_textures(out_textures, count))
    {
        return 0;
    }

    std::vector<VkDescriptorImageInfo> image_info(count);

    for (size_t i = 0; i < count; ++i)
    {
        VkDescriptorImageInfo& ii = image_info[i];
        ii.imageLayout = out_textures[i].layout;
        ii.imageView = out_textures[i].view;
    }

    VkWriteDescriptorSet desc_write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    desc_write.dstSet = ctx.bindless_descriptor_set;
    desc_write.dstBinding = 0;
    desc_write.dstArrayElement = 0;
    desc_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    desc_write.descriptorCount = count;
    desc_write.pImageInfo = image_info.data();
    vkUpdateDescriptorSets(ctx.device, 1, &desc_write, 0, nullptr);

    return count;
}

size_t load_materials(Context& ctx, const cgltf_data* gltf_data, Material* out_materials, size_t count)
{
    if (out_materials == nullptr) return gltf_data->materials_count;

    for (size_t i = 0; i < count; ++i)
    {
        const cgltf_material& mat = gltf_data->materials[i];

        Material& m = out_materials[i];
        assert(mat.has_pbr_metallic_roughness && "Unsupported material type!");

        m.basecolor_factor = glm::make_vec4(mat.pbr_metallic_roughness.base_color_factor);
        m.roughness_factor = mat.pbr_metallic_roughness.roughness_factor;
        m.metallic_factor = mat.pbr_metallic_roughness.metallic_factor;

        m.basecolor_texture = mat.pbr_metallic_roughness.base_color_texture.texture ? cgltf_texture_index(gltf_data, mat.pbr_metallic_roughness.base_color_texture.texture) : -1;
        m.metallic_roughness_texture = mat.pbr_metallic_roughness.metallic_roughness_texture.texture ? cgltf_texture_index(gltf_data, mat.pbr_metallic_roughness.metallic_roughness_texture.texture) : -1;
        m.normal_texture = mat.normal_texture.texture ? cgltf_texture_index(gltf_data, mat.normal_texture.texture) : -1;

        switch (mat.alpha_mode)
        {
        case cgltf_alpha_mode_opaque:
            m.alpha_cutoff = 1.0f;
            break;
        case cgltf_alpha_mode_mask:
            m.alpha_cutoff = mat.alpha_cutoff;
            break;
        case cgltf_alpha_mode_blend:
            LOG_WARNING("Unimplemented alpha mode: Alpha blend");
            break;
        }
    }

    return count;
}
