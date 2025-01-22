#pragma once

#include "defines.h"
#include "pipeline.h"

#include <set>
#include <filesystem>

struct IAsset
{
	virtual const std::set<std::filesystem::path>& get_dependencies() const = 0;
	virtual bool reload_asset() = 0;
	virtual size_t get_hash() = 0;
};

struct GraphicsPipelineAsset : IAsset
{
	virtual const std::set<std::filesystem::path>& get_dependencies() const override;
	virtual bool reload_asset() override;
	virtual size_t get_hash() override;

	GraphicsPipelineAsset(GraphicsPipelineBuilder build);

	std::set<std::filesystem::path> dependencies;
	GraphicsPipelineBuilder builder;
	Pipeline pipeline;
};

struct ComputePipelineAsset : IAsset
{
	virtual const std::set<std::filesystem::path>& get_dependencies() const override;
	virtual bool reload_asset() override;
	virtual size_t get_hash() override;

	ComputePipelineAsset(ComputePipelineBuilder build);

	ComputePipelineBuilder builder;
	Pipeline pipeline;
};

namespace AssetCatalog
{
	void register_asset(IAsset* asset);
	bool check_for_dirty_assets();
	bool reload_dirty_assets();
	void force_reload_all();
}