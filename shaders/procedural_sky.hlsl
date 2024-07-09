[[vk::binding(0)]] cbuffer globals {
    float4x4 view;
    float4x4 projection;
    float4x4 viewprojection;
    float4 camera_pos;
    float4 sun_direction;
    float4 sun_color_and_intensity;
}

[[vk::binding(1)]] RWTexture2D<float4> out_texture;

[numthreads(8, 8, 1)]
void cs_main( uint3 thread_id : SV_DispatchThreadID )
{
    uint w, h;
    out_texture.GetDimensions(w, h);
    float2 uv = float2(thread_id.xy + 0.5) / float2(w, h);
    if (all(saturate(uv) == uv))
    {
        out_texture[thread_id.xy] = float4(1.0, 1.0, 1.0, 1.0);
    }
}