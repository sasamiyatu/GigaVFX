#include "shaders.h"

#ifdef _WIN32
#include "windows.h"
#include <atlbase.h>
#else _WIN32
#include <dxc/WinAdapter.h>
#endif

#include <dxc/dxcapi.h>
#include <filesystem>
#include <map>

#include "misc.h"

#define OPTIMIZE_SHADERS 1

namespace
{
	static CComPtr<IDxcUtils> dxc_utils;
	static CComPtr<IDxcCompiler3> compiler;
	static CComPtr<IDxcIncludeHandler> include_handler;

	static bool initialized = false;
}

static LPWSTR get_entry_point(VkShaderStageFlagBits shader_stage)
{
	switch (shader_stage)
	{
	case VK_SHADER_STAGE_VERTEX_BIT:
		return L"vs_main";
	case VK_SHADER_STAGE_FRAGMENT_BIT:
		return L"fs_main";
	case VK_SHADER_STAGE_COMPUTE_BIT:
		return L"cs_main";
	default:
		assert(false);
		return nullptr;
	}
}

static LPWSTR get_shader_type_str(VkShaderStageFlagBits shader_stage)
{
	switch (shader_stage)
	{
	case VK_SHADER_STAGE_VERTEX_BIT:
		return L"vs_6_6";
	case VK_SHADER_STAGE_FRAGMENT_BIT:
		return L"ps_6_6";
	case VK_SHADER_STAGE_COMPUTE_BIT:
		return L"cs_6_6";
	default:
		assert(false);
		return nullptr;
	}
}

namespace Shaders
{

void init()
{
	DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils));
	DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

	dxc_utils->CreateDefaultIncludeHandler(&include_handler);

	initialized = true;
}

uint32_t* Shaders::load_shader(const char* filepath, const char* entry_point, VkShaderStageFlagBits shader_stage, uint32_t* size)
{
#if 0 
	assert(initialized);

	auto cwd = std::filesystem::current_path(); // Current working directory
	std::filesystem::current_path(cwd / std::filesystem::path(std::string("shaders")));

	std::string shader_src = read_text_file(filepath);
	if (shader_src.empty()) return nullptr;

	DxcBuffer src{};
	src.Ptr = shader_src.data();
	src.Size = shader_src.length();
	src.Encoding = DXC_CP_ACP;

	std::wstring ep_str;
	if (entry_point) ep_str = std::wstring(entry_point, entry_point + strlen(entry_point));
	LPCWSTR args[] = {
		L"-E", !ep_str.empty() ? ep_str.data() : get_entry_point(shader_stage),
		L"-T", get_shader_type_str(shader_stage),
		L"-Zs", L"-spirv",
		L"-fvk-use-scalar-layout",
		L"-fspv-target-env=vulkan1.3",
		L"-HV 2021",
#if OPTIMIZE_SHADERS
		L"-O3"
#else
		L"-O0"
#endif
	};

	CComPtr<IDxcResult> results;
	compiler->Compile(&src, args, _countof(args), include_handler, IID_PPV_ARGS(&results));

	CComPtr<IDxcBlobUtf8> errors = nullptr;
	results->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
	// Note that d3dcompiler would return null if no errors or warnings are present.
	// IDxcCompiler3::Compile will always return an error buffer, but its length
	// will be zero if there are no warnings or errors.
	if (errors != nullptr && errors->GetStringLength() != 0)
		LOG_ERROR("Shader compilation warnings/errors: %s", errors->GetStringPointer());

	HRESULT hrStatus;
	results->GetStatus(&hrStatus);
	if (FAILED(hrStatus))
	{
		LOG_ERROR("Shader Compilation Failed");
		std::filesystem::current_path(cwd);
		return nullptr;
	}

	CComPtr<IDxcBlob> shader = nullptr;
	CComPtr<IDxcBlobUtf16> shader_name = nullptr;
	results->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader), &shader_name);

	uint32_t* data = (uint32_t*)malloc(shader->GetBufferSize());
	assert(data);
	memcpy(data, shader->GetBufferPointer(), shader->GetBufferSize());

	*size = shader->GetBufferSize();

	std::filesystem::current_path(cwd);

	return data;
#endif
	ShaderSource source(filepath, entry_point);

	return load_shader(source, shader_stage, size);
}

uint32_t* load_shader(const ShaderSource& shader_source, VkShaderStageFlagBits shader_stage, uint32_t* size)
{
	assert(initialized);

	auto cwd = std::filesystem::current_path(); // Current working directory
	std::filesystem::current_path(cwd / std::filesystem::path(std::string("shaders")));

	std::string shader_src = read_text_file(shader_source.filepath.c_str());
	if (shader_src.empty()) return nullptr;

	for (const auto& d : shader_source.defines)
	{
		shader_src.insert(0, "#define " + d.first + " " + d.second + "\n");
	}
	// Add defines

	DxcBuffer src{};
	src.Ptr = shader_src.data();
	src.Size = shader_src.length();
	src.Encoding = DXC_CP_ACP;

	std::wstring ep_str;
	if (!shader_source.entry_point.empty()) ep_str = std::wstring(shader_source.entry_point.begin(), shader_source.entry_point.end());
	LPCWSTR args[] = {
		L"-E", !ep_str.empty() ? ep_str.data() : get_entry_point(shader_stage),
		L"-T", get_shader_type_str(shader_stage),
		L"-Zs", L"-spirv",
		L"-fvk-use-scalar-layout",
		L"-fspv-target-env=vulkan1.3",
		L"-HV 2021",
#if OPTIMIZE_SHADERS
		L"-O3"
#else
		L"-O0"
#endif
	};

	CComPtr<IDxcResult> results;
	compiler->Compile(&src, args, _countof(args), include_handler, IID_PPV_ARGS(&results));

	CComPtr<IDxcBlobUtf8> errors = nullptr;
	results->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
	// Note that d3dcompiler would return null if no errors or warnings are present.
	// IDxcCompiler3::Compile will always return an error buffer, but its length
	// will be zero if there are no warnings or errors.
	if (errors != nullptr && errors->GetStringLength() != 0)
		LOG_ERROR("Shader compilation warnings/errors: %s", errors->GetStringPointer());

	HRESULT hrStatus;
	results->GetStatus(&hrStatus);
	if (FAILED(hrStatus))
	{
		LOG_ERROR("Shader Compilation Failed");
		std::filesystem::current_path(cwd);
		return nullptr;
	}

	CComPtr<IDxcBlob> shader = nullptr;
	CComPtr<IDxcBlobUtf16> shader_name = nullptr;
	results->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader), &shader_name);

	uint32_t* data = (uint32_t*)malloc(shader->GetBufferSize());
	assert(data);
	memcpy(data, shader->GetBufferPointer(), shader->GetBufferSize());

	*size = shader->GetBufferSize();

	std::filesystem::current_path(cwd);

	return data;
}

} // namespace Shaders

void ShaderSource::add_defines(const std::string& first, const std::string& second)
{
	if (defines.count(first) != 0)
	{
		LOG_WARNING("Defines for shader %s, entry point %s already contains '%s'! Overwriting...",
			filepath.c_str(), entry_point.c_str(), first.c_str()
		);
	}

	defines.insert(std::make_pair(first, second));
}

void ShaderSource::add_defines(const std::string& str)
{
	add_defines(str, "");
}
