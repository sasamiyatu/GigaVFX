[[vk::binding(0)]] RWTexture2D<float4> particle_texture;
[[vk::binding(1)]] RWTexture2D<float4> out_render_target;

[numthreads(8, 8, 1)]
void cs_composite_image( uint3 thread_id : SV_DispatchThreadID )
{
    int w, h;
    particle_texture.GetDimensions(w, h);

    if (thread_id.x >= w || thread_id.y >= h) return;

    float4 particle_val = particle_texture[thread_id.xy];
    float4 rt_val = out_render_target[thread_id.xy];

    float4 composite = particle_val + rt_val * (1.0 - particle_val.a);
    out_render_target[thread_id.xy] = composite;
}
