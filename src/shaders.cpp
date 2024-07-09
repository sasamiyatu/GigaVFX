#include "shaders.h"

#ifdef _WIN32
#include "windows.h"
#include <atlbase.h>
#else _WIN32
#include <dxc/WinAdapter.h>
#endif

#include <dxc/dxcapi.h>
#include <filesystem>

#include "misc.h"

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

uint32_t* Shaders::load_shader(const char* filepath, VkShaderStageFlagBits shader_stage, uint32_t* size)
{
	assert(initialized);

	auto cwd = std::filesystem::current_path(); // Current working directory
	std::filesystem::current_path(cwd / std::filesystem::path(std::string("shaders")));

	size_t file_size = 0;
	uint8_t* bytes = read_entire_file(filepath, &file_size);
	if (!bytes) return nullptr;

	DxcBuffer src{};
	src.Ptr = bytes;
	src.Size = file_size;
	src.Encoding = DXC_CP_ACP;

	LPCWSTR args[] = {
		L"-E", get_entry_point(shader_stage),
		L"-T", get_shader_type_str(shader_stage),
		L"-Zs", L"-spirv",
		L"-HV 2021",
		L"-O0"
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

	free(bytes);

	std::filesystem::current_path(cwd);

	return data;
}

} // namespace Shaders