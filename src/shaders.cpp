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
#include <sstream>
#include <set>

#include "misc.h"

#define OPTIMIZE_SHADERS 1

namespace
{
	static CComPtr<IDxcUtils> dxc_utils;
	static CComPtr<IDxcCompiler3> compiler;
	//static CComPtr<IDxcIncludeHandler> include_handler;

	struct MyIncludeHandler : IDxcIncludeHandler
	{
		HRESULT LoadSource(
			LPCWSTR  pFilename,
			IDxcBlob** ppIncludeSource)
		{
			CComPtr<IDxcBlobEncoding> pEncoding;
			std::wstring wstr(pFilename);
			std::string path(wstr.begin(), wstr.end());
			std::filesystem::path p(std::filesystem::path("shaders") / std::filesystem::path(path));
			auto cwd = std::filesystem::current_path(); // Current working directory
			if (included_files.find(path) != included_files.end())
			{
				// Return empty string blob if this file has been included before
				static const char nullStr[] = " ";
				dxc_utils->CreateBlobFromPinned(nullStr, ARRAYSIZE(nullStr), DXC_CP_ACP, &pEncoding);
				*ppIncludeSource = pEncoding.Detach();
				return S_OK;
			}

			//HRESULT hr = dxc_utils->LoadFile(pFilename, nullptr, &pEncoding);
			HRESULT hr = dxc_utils->LoadFile(pFilename, nullptr, &pEncoding);
			if (SUCCEEDED(hr))
			{
				included_files.insert(path);
				*ppIncludeSource = pEncoding.Detach();
			}
			else
			{
				LOG_ERROR("Failed to load shader include files '%s'", pFilename);
			}
			
			return hr;
		}

		HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, _COM_Outptr_ void __RPC_FAR* __RPC_FAR* ppvObject) override { return E_NOINTERFACE; }
		ULONG STDMETHODCALLTYPE AddRef(void) override { return 0; }
		ULONG STDMETHODCALLTYPE Release(void) override { return 0; }

		std::set<std::string> included_files;
	};

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

static std::string load_shader_source(ShaderSource& src)
{
	std::string shader_src = read_text_file(src.filepath.c_str());
	if (shader_src.empty()) return "";

	// Make sure file ends in a new line symbol
	if (shader_src.back() != '\n')
		shader_src.push_back('\n');

	for (const auto& l : src.prepend_lines)
	{
		shader_src.insert(0, l + "\n");
	}

	for (const auto& l : src.append_lines)
	{
		shader_src.insert(shader_src.length(), l + "\n");
	}

	return shader_src;
}

namespace Shaders
{

void init()
{
	DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils));
	DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

	//dxc_utils->CreateDefaultIncludeHandler(&include_handler);

	initialized = true;
}

uint32_t* Shaders::load_shader(const char* filepath, const char* entry_point, VkShaderStageFlagBits shader_stage, uint32_t* size)
{
	ShaderSource source(filepath, entry_point);

	return load_shader(source, shader_stage, size);
}

uint32_t* load_shader(ShaderSource& shader_source, VkShaderStageFlagBits shader_stage, uint32_t* size)
{
	assert(initialized);

	auto cwd = std::filesystem::current_path(); // Current working directory
	std::filesystem::current_path(cwd / std::filesystem::path(std::string("shaders")));

	std::string shader_src = load_shader_source(shader_source);
	if (shader_src.empty()) return nullptr;

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

	MyIncludeHandler include_handler;

	CComPtr<IDxcResult> results;
	compiler->Compile(&src, args, _countof(args), &include_handler, IID_PPV_ARGS(&results));

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

	auto dir = std::filesystem::current_path();
	shader_source.dependencies = { dir / std::filesystem::path(shader_source.filepath) };
	for (const auto& d : include_handler.included_files)
	{
		auto p = dir / std::filesystem::relative(d);
		shader_source.dependencies.insert(p);
	}

	LOG_DEBUG("Shader source (f: '%s', ep: '%s' has dependencies:", shader_source.filepath.c_str(), shader_source.entry_point.c_str());
	for (const auto& d : shader_source.dependencies)
	{
		LOG_DEBUG("\t%s", d.string().c_str());
	}

	std::filesystem::current_path(cwd);

	return data;
}

} // namespace Shaders

void ShaderSource::add_defines(const std::string& first, const std::string& second)
{
	prepend_lines.push_back("#define " + first + " " + second);
}

void ShaderSource::add_defines(const std::string& str)
{
	prepend_lines.push_back("#define " + str);
}

void ShaderSource::add_include(const std::string& str, bool append)
{
	auto& v = append ? append_lines : prepend_lines;
	v.push_back("#include \"" + str + "\"");
}

void ShaderSource::add_specialization_constant(uint32_t constant_id, bool value)
{
	SpecializationConstantEntry entry{};
	entry.type = SpecializationConstantEntry::BOOL;
	entry.constant_id = constant_id;
	entry.bool_val = value;
	specialization_constants.push_back(entry);
}

void ShaderSource::add_specialization_constant(uint32_t constant_id, uint32_t value)
{
	SpecializationConstantEntry entry{};
	entry.type = SpecializationConstantEntry::UINT;
	entry.constant_id = constant_id;
	entry.uint_val = value;
	specialization_constants.push_back(entry);
}

void ShaderSource::add_specialization_constant(uint32_t constant_id, float value)
{
	SpecializationConstantEntry entry{};
	entry.type = SpecializationConstantEntry::FLOAT;
	entry.constant_id = constant_id;
	entry.float_val = value;
	specialization_constants.push_back(entry);
}

