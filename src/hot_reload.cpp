#include "hot_reload.h"
#include <unordered_map>
#include "timer.h"

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

const std::set<std::filesystem::path>& GraphicsPipelineAsset::get_dependencies() const
{
	return dependencies;
}

bool GraphicsPipelineAsset::reload_asset()
{
	bool result = builder.build(&pipeline);
	if (result)
	{
		dependencies.clear();
		for (uint32_t i = 0; i < builder.pipeline_create_info.stageCount; ++i)
		{
			dependencies.insert(
				builder.shader_sources[i].shader_source.dependencies.begin(),
				builder.shader_sources[i].shader_source.dependencies.end()
			);
		}
	}

	return result;
}

size_t GraphicsPipelineAsset::get_hash()
{
	return murmur3_32((uint8_t*)this, sizeof(GraphicsPipelineAsset), 42);
}

GraphicsPipelineAsset::GraphicsPipelineAsset(GraphicsPipelineBuilder build)
	: builder(build)
{
	if (!reload_asset())
	{
		assert(false);
		exit(1);
	}
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
	uint64_t timestamp = 0;
	const auto& dependencies = asset->get_dependencies();
	for (const auto& d : dependencies)
	{
		timestamp = std::max(get_file_timestamp(d, ec), timestamp);
	}
	new_asset.last_file_write = timestamp;
	registered_assets.push_back(new_asset);
}

bool AssetCatalog::check_for_dirty_assets()
{
	Timer timer;
	timer.tick();

	bool any_dirty = false;

	for (auto& asset : registered_assets)
	{
		const auto& dependencies = asset.asset->get_dependencies();
		assert(!dependencies.empty() && "Logic error: Asset has no dependencies!");
		std::error_code ec;
		uint64_t timestamp = 0;
		for (const auto& d : dependencies)
		{
			timestamp = std::max(get_file_timestamp(d, ec), timestamp);
		}
		if (timestamp != asset.last_file_write)
		{
			asset.dirty = true;
			LOG_INFO("Asset has been updated!");
		}

		any_dirty = any_dirty || asset.dirty;
	}

	timer.tock();
	//LOG_DEBUG("Checking for dirty assets took %f ms", timer.get_elapsed_milliseconds());

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
				uint64_t timestamp = 0;
				const auto& dependencies = a.asset->get_dependencies();
				for (const auto& d : dependencies)
				{
					timestamp = std::max(get_file_timestamp(d, ec), timestamp);
				}
				a.last_file_write = timestamp;
				LOG_INFO("Successfully reloaded asset");
			}
			else
			{
				LOG_ERROR("Failed to reload asset");
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

const std::set<std::filesystem::path>& ComputePipelineAsset::get_dependencies()  const
{
	return builder.shader_source.shader_source.dependencies;
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
	if (!reload_asset())
		exit(1);
}
