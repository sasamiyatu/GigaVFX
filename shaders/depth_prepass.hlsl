#include "shared.h"

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

struct VSInput
{
    uint vertex_id: SV_VertexID;
};

struct VSOutput
{
    float4 position: SV_Position;
};

[[vk::push_constant]]
PushConstantsForward push_constants;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 pos = vk::RawBufferLoad<float3>(push_constants.position_buffer + input.vertex_id * 12);
    float3 world_pos = mul(push_constants.model, float4(pos, 1.0)).xyz;
    
    output.position = mul(globals.viewprojection, float4(world_pos, 1.0));

    return output;
}

void fs_main(VSOutput input)
{
}