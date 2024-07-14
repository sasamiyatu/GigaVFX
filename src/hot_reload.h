#pragma once

#include "defines.h"
#include <filesystem>
#include "pipeline.h"

struct IAsset
{
	virtual std::filesystem::path get_filepath() const = 0;
	virtual bool reload_asset() = 0;
};

struct GraphicsPipelineAsset : IAsset
{
	virtual std::filesystem::path get_filepath() const override;
	virtual bool reload_asset() override;

	GraphicsPipelineAsset(GraphicsPipelineBuilder build);

	std::string shader_path;
	GraphicsPipelineBuilder builder;
	Pipeline pipeline;
};

struct ComputePipelineAsset : IAsset
{
	virtual std::filesystem::path get_filepath() const override;
	virtual bool reload_asset() override;

	ComputePipelineAsset(ComputePipelineBuilder build);

	std::string shader_path;
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