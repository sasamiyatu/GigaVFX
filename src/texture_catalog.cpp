#include "texture_catalog.h"
#include <filesystem>
#include <assert.h>
#include "graphics_context.h"
#include "imgui/imgui.h"

void TextureCatalog::init(Context* ctx, const char* texture_directory)
{
	directory = texture_directory;
	context = ctx;

	std::filesystem::path path = directory;
	assert(std::filesystem::exists(path));

	for (const auto& f : std::filesystem::directory_iterator(path))
	{
		if (f.is_regular_file())
		{
			std::filesystem::path extension = f.path().extension();
			if (extension == ".tif") continue;
			printf("%s\n", f.path().string().c_str());
			std::string name = f.path().filename().string();
			Texture texture{};
			if (!load_texture_from_file(f.path().string().c_str(), texture))
			{
				LOG_ERROR("Failed to load texture!");
				exit(EXIT_FAILURE);
			}

			ctx->create_textures(&texture, 1);
			textures.insert(std::make_pair(f.path().string(), texture));
		}
	}
}

void TextureCatalog::shutdown()
{
	for (auto& t : textures)
	{
		t.second.destroy(context->device, context->allocator);
	}

	textures.clear();
}

void TextureCatalog::draw_ui(bool *open)
{
	ImGuiIO& io = ImGui::GetIO();
	ImGui::Begin("Texture browser", open);

	static int selected = -1;
	int index = 0;
	static const Texture* texture = nullptr;
	static bool texture_preview_open = false;
	bool changed = false;
	for (const auto& t : textures)
	{
		if (ImGui::Selectable(t.first.c_str(), selected == index))
		{
			selected = index;
			changed = &t.second != texture;
			texture = &t.second;
		}

		if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
		{
			texture_preview_open = true;
		}

		++index;
	}

	if (texture && texture_preview_open)
	{
		static float zoom = 1.0f;
		constexpr float zoom_step = 0.10f;
		static glm::vec2 uv0 = glm::vec2(0.0f, 0.0f);

		if (changed)
		{
			zoom = 1.0f;
			uv0 = glm::vec2(0.0f);
		}

		constexpr glm::vec2 window_size = glm::vec2(1024.0f, 1024.0f);
		glm::vec2 unscaled_uv1 = window_size / glm::vec2(texture->width, texture->height);
		glm::vec2 uv1 = uv0 + unscaled_uv1 * zoom;
		ImGui::SetNextWindowBgAlpha(1.0f);
		ImGui::Begin("Texture preview", &texture_preview_open, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_AlwaysAutoResize);
		ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
		ImGui::Image(texture->descriptor_set, ImVec2(window_size.x, window_size.y), ImVec2(uv0.x, uv0.y), ImVec2(uv1.x, uv1.y));
		if (ImGui::IsItemHovered())
		{
			if (ImGui::IsMouseDragging(0))
			{
				ImVec2 drag_delta = ImGui::GetMouseDragDelta(0);
				ImGui::ResetMouseDragDelta(0);
				uv0 -= glm::vec2((float)drag_delta.x / texture->width, (float)drag_delta.y / texture->height);
			}
			float wheel = io.MouseWheel;
			if (wheel != 0.0f)
			{
				ImVec2 relative_pos;
				relative_pos.x = ImGui::GetMousePos().x - cursor_pos.x;
				relative_pos.y = ImGui::GetMousePos().y - cursor_pos.y;

				glm::vec2 uv_before = glm::vec2(relative_pos.x, relative_pos.y) / glm::vec2(texture->width, texture->height) * zoom + uv0;
				zoom -= zoom_step * wheel;
				zoom = glm::max(0.0f, zoom);
				glm::vec2 uv_after = glm::vec2(relative_pos.x, relative_pos.y) / glm::vec2(texture->width, texture->height) * zoom + uv0;
				glm::vec2 delta = uv_after - uv_before;
				uv0 -= delta;
			}

		}
		ImGui::End();
	}

	ImGui::End();
}

Texture* TextureCatalog::get_texture(const char* name)
{
	if (textures.count(name) != 0)
		return &textures[name];

	return nullptr;
}
