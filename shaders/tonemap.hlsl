#include "shared.h"
#include "atmosphere.hlsli"
#include "misc.hlsli"


[[vk::binding(0)]] RWTexture2D<float4> in_texture;
[[vk::binding(1)]] RWTexture2D<float4> out_texture;

[[vk::push_constant]]
PushConstantsTonemap push_constants;

float3 ACESFilm(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

[numthreads(8, 8, 1)]
void cs_main( uint3 thread_id : SV_DispatchThreadID )
{
    if (all(thread_id.xy < push_constants.size))
    {
        float4 color = in_texture[thread_id.xy];
        color.rgb = ACESFilm(color.rgb);
        color.rgb = linear_to_srgb(color.rgb);
        out_texture[thread_id.xy] = float4(color.rgb, 1.0);
    }
}