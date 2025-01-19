#include "hot_reload.h"
#include <unordered_map>

static inline uint32_t murmur_32_scramble(uint32_t k) {
	k *= 0xcc9e2d51;
	k = (k << 15) | (k >> 17);
	k *= 0x1b873593;
	return k;
}

uint32_t murmur3_32(const uint8_t* key, size_t len, uint32_t seed)
{
	uint32_t h = seed;
	uint32_t k;
	/* Read in groups of 4. */
	for (size_t i = len >> 2; i; i--) {
		// Here is a source of differing results across endiannesses.
		// A swap here has no effects on hash properties though.
		memcpy(&k, key, sizeof(uint32_t));
		key += sizeof(uint32_t);
		h ^= murmur_32_scramble(k);
		h = (h << 13) | (h >> 19);
		h = h * 5 + 0xe6546b64;
	}
	/* Read the rest. */
	k = 0;
	for (size_t i = len & 3; i; i--) {
		k <<= 8;
		k |= key[i - 1];
	}
	// A swap is *not* necessary here because the preceding loop already
	// places the low bytes in the low places according to whatever endianness
	// we use. Swaps only apply when the memory is copied in a chunk.
	h ^= murmur_32_scramble(k);
	/* Finalize. */
	h ^= len;
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h;
}

std::filesystem::path GraphicsPipelineAsset::get_filepath() const
{
	return std::filesystem::path(std::string("shaders")) / std::filesystem::path(shader_path);
}

bool GraphicsPipelineAsset::reload_asset()
{
	return builder.build(&pipeline);
}

size_t GraphicsPipelineAsset::get_hash()
{
	return murmur3_32((uint8_t*)this, sizeof(GraphicsPipelineAsset), 42);
}

GraphicsPipelineAsset::GraphicsPipelineAsset(GraphicsPipelineBuilder build)
	: builder(build)
{
	if (!builder.build(&pipeline))
	{
		assert(false);
		exit(1);
	}

	shader_path = builder.shader_sources[0].shader_source.filepath;
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
			return asset->get_hash() == a.asset->get_hash();
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

		any_dirty = any_dirty || asset.dirty;
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
				all_success = false;
			}
		}
	}
	return all_success;
}

void AssetCatalog::force_reload_all()
{
	for (auto& a : registered_assets) a.dirty = true;
}

std::filesystem::path ComputePipelineAsset::get_filepath() const
{
	return std::filesystem::path(std::string("shaders")) / std::filesystem::path(shader_path);
}

bool ComputePipelineAsset::reload_asset()
{
	return builder.build(&pipeline);
}

size_t ComputePipelineAsset::get_hash()
{
	return murmur3_32((uint8_t*)this, sizeof(ComputePipelineAsset), 1337);
}

ComputePipelineAsset::ComputePipelineAsset(ComputePipelineBuilder build)
	: builder(build)
{
	if (!builder.build(&pipeline))
		exit(1);

	shader_path = builder.shader_source.shader_source.filepath;
}
