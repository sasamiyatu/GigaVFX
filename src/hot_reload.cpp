#include "hot_reload.h"
#include <unordered_map>

std::filesystem::path GraphicsPipelineAsset::get_filepath() const
{
	return std::filesystem::path(std::string("shaders")) / std::filesystem::path(shader_path);
}

bool GraphicsPipelineAsset::reload_asset()
{
	return builder.build(&pipeline);
}

GraphicsPipelineAsset::GraphicsPipelineAsset(GraphicsPipelineBuilder build)
	: builder(build)
{
	if (!builder.build(&pipeline))
		exit(1);

	shader_path = builder.shader_sources[0].filepath;
}

struct RegisteredAsset
{
	IAsset* asset;
	uint64_t last_file_write;
	bool dirty = false;
};

static uint64_t get_file_timestamp(const std::filesystem::path& path, std::error_code& ec)
{
	std::filesystem::file_time_type ftime = std::filesystem::last_write_time(path, ec);
	if (ec.value() != 0)
	{
		LOG_ERROR("Get file timestamp failed: %s", ec.message().c_str());
	}
	return ftime.time_since_epoch().count();
}

static std::vector<RegisteredAsset> registered_assets;

void AssetCatalog::register_asset(IAsset* asset)
{
	auto iter = std::find_if(registered_assets.begin(), registered_assets.end(), [=](const RegisteredAsset& a)
		{
			return asset->get_filepath() == a.asset->get_filepath();
		});

	if (iter != registered_assets.end())
	{
		LOG_ERROR("Asset already registered!");
		return;
	}

	RegisteredAsset new_asset{};
	new_asset.asset = asset;
	std::error_code ec;
	new_asset.last_file_write = get_file_timestamp(asset->get_filepath(), ec);
	registered_assets.push_back(new_asset);
}

bool AssetCatalog::check_for_dirty_assets()
{
	bool any_dirty = false;
	for (auto& asset : registered_assets)
	{
		auto filepath = asset.asset->get_filepath();
		std::error_code ec;
		auto timestamp = get_file_timestamp(filepath, ec);
		if (timestamp != asset.last_file_write)
		{
			asset.dirty = true;
			LOG_INFO("Asset %s has been updated!", filepath.string().c_str());
		}

		any_dirty = asset.dirty;
	}
	return any_dirty;
}

bool AssetCatalog::reload_dirty_assets()
{
	bool all_success = true;
	for (auto& a : registered_assets)
	{
		if (a.dirty)
		{
			if (a.asset->reload_asset())
			{
				a.dirty = false;

				std::error_code ec;
				a.last_file_write = get_file_timestamp(a.asset->get_filepath(), ec);
				LOG_INFO("Successfully reloaded asset %s", a.asset->get_filepath().string().c_str());
			}
			else
			{
				LOG_ERROR("Failed to reload asset %s", a.asset->get_filepath().string().c_str());
			}
		}
	}
	return all_success;
}

std::filesystem::path ComputePipelineAsset::get_filepath() const
{
	return std::filesystem::path(std::string("shaders")) / std::filesystem::path(shader_path);
}

bool ComputePipelineAsset::reload_asset()
{
	return builder.build(&pipeline);
}

ComputePipelineAsset::ComputePipelineAsset(ComputePipelineBuilder build)
	: builder(build)
{
	if (!builder.build(&pipeline))
		exit(1);

	shader_path = builder.shader_source.filepath;
}
