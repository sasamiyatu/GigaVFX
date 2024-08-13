#pragma once
#include <map>
#include <string>

struct Context;
struct Texture;

struct TextureCatalog
{
	void init(Context* context, const char* texture_directory);
	void shutdown();
	void draw_ui(bool* open);
	Texture* get_texture(const char* name);

	const char* directory;

	std::map<std::string, Texture> textures;
	Context* context;
};