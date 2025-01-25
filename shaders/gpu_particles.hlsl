#include "shared.h"
#include "math.hlsli"
#include "random.hlsli"
#include "noise.hlsli"
#include "misc.hlsli"
#include "color.hlsli"

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] cbuffer system_globals
{
    GPUParticleSystemGlobals system_globals;
};

[[vk::binding(2)]] RWStructuredBuffer<GPUParticle> particles;

[[vk::binding(3)]] RWStructuredBuffer<GPUParticleSystemState> particle_system_state;

[[vk::binding(4)]] RWStructuredBuffer<GPUParticle> particles_compact_out;
[[vk::binding(5)]] RWStructuredBuffer<GPUParticleSystemState> particle_system_state_out;

[[vk::binding(6)]] RWStructuredBuffer<DispatchIndirectCommand> indirect_dispatch;
[[vk::binding(7)]] RWStructuredBuffer<GPUParticleSort> particle_sort;
[[vk::binding(8)]] RWStructuredBuffer<AABBPositions> aabb_positions;
[[vk::binding(9)]] RWStructuredBuffer<AccelerationStructureInstance> instances;
[[vk::binding(10)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(11)]] RWStructuredBuffer<DrawIndirectCommand> indirect_draw;
[[vk::binding(12)]] SamplerState light_sampler;
[[vk::binding(13)]] Texture2D light_texture;

[[vk::push_constant]]
GPUParticlePushConstants push_constants;

[numthreads(64, 1, 1)]
void cs_emit_particles( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= push_constants.particles_to_spawn)
        return;

    if (particle_system_state[0].active_particle_count >= system_globals.particle_capacity)
        return;

    uint lane_spawn_count = WaveActiveCountBits(true); // == Number of threads that passed the early return checks
    uint local_index = WaveGetLaneIndex();
    uint global_particle_index;

    if (WaveIsFirstLane())
        InterlockedAdd(particle_system_state[0].active_particle_count, lane_spawn_count, global_particle_index);

    global_particle_index = WaveReadLaneFirst(global_particle_index);

    uint4 seed = uint4(thread_id.x, globals.frame_index, 42, 1337);
    float3 point_on_sphere = sample_uniform_sphere(uniform_random(seed).xy);
    GPUParticle p;
    p.velocity = 0;
    p.lifetime = push_constants.lifetime;
    p.position = push_constants.smoke_origin + point_on_sphere * push_constants.emitter_radius * sqrt(float(seed.z) / float(0xFFFFFFFFu));
    p.color = float4(hsv2rgb(float3(frac(globals.time * 0.02), 0.5, 0.5)), push_constants.particle_color.a);
    particles[global_particle_index + local_index] = p;
}

[numthreads(64, 1, 1)]
void cs_simulate_particles( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    GPUParticle p = particles[thread_id.x];
    if (p.lifetime > 0.0)
    {
        float age = push_constants.lifetime - p.lifetime;
        p.color.a = smoothstep(0.0, 0.2, age) * push_constants.particle_color.a;
        float age_scale = smoothstep(0.0, push_constants.lifetime, age) * 2.0;
        p.velocity = (push_constants.smoke_dir * 3.0 + curl_noise(p.position, push_constants.time * push_constants.noise_time_scale) * (1 + age_scale) * push_constants.noise_scale) * push_constants.speed;
        p.position += p.velocity * push_constants.delta_time;
        p.lifetime -= push_constants.delta_time;
    }

    particles[thread_id.x] = p;
}


[numthreads(1, 1, 1)]
void cs_write_dispatch( uint3 thread_id : SV_DispatchThreadID )
{
    uint size = (particle_system_state[0].active_particle_count + 63) / 64;

    DispatchIndirectCommand command;
    command.x = size;
    command.y = 1;
    command.z = 1;

    indirect_dispatch[0] = command;
}

[numthreads(64, 1, 1)]
void cs_write_draw( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= push_constants.num_slices) return;

    uint active_particles = particle_system_state_out[0].active_particle_count;
    uint draw_count = push_constants.num_slices == 0 
        ? active_particles
        : (active_particles + push_constants.num_slices - 1) / push_constants.num_slices;

    uint32_t total_draws = active_particles;
    uint32_t draws_per_slice = (total_draws + push_constants.num_slices - 1) / push_constants.num_slices;
    uint draws_left = max(0, int(total_draws) - int(draws_per_slice * thread_id.x));

    DrawIndirectCommand draw_cmd;
    draw_cmd.vertexCount = 1;
    draw_cmd.instanceCount = min(draws_left, draws_per_slice);
    draw_cmd.firstVertex = 0;
    draw_cmd.firstInstance = draw_count * thread_id.x;

    indirect_draw[thread_id.x] = draw_cmd;
}

// TODO: Should probably be merged with simulate
[numthreads(64, 1, 1)]
void cs_compact_particles( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    GPUParticle p = particles[thread_id.x];

    bool alive = p.lifetime > 0.0;
    uint local_index = WavePrefixCountBits(alive);
    uint alive_count = WaveActiveCountBits(alive);

    if (alive_count == 0) return;

    uint global_particle_index;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(particle_system_state_out[0].active_particle_count, alive_count, global_particle_index);
    }

    global_particle_index = WaveReadLaneFirst(global_particle_index);

    if (alive)
    {
        uint index = global_particle_index + local_index;
        particles_compact_out[index] = p;

        float projected = dot(push_constants.sort_axis, p.position);
        GPUParticleSort sort;
        sort.index = index;
        sort.key = sort_key_from_float(asuint(projected));

        particle_sort[index] = sort;
    }
}

// Used for debugging only
[numthreads(1, 1, 1)]
void cs_debug_print_sorted_particles( uint3 thread_id : SV_DispatchThreadID )
{
    uint count = particle_system_state_out[0].active_particle_count;
    printf("active count %d, sort_axis: (%f, %f, %f)\n", count, 
        push_constants.sort_axis.x,
        push_constants.sort_axis.y,
        push_constants.sort_axis.z);

    printf("Unsorted");
    for (uint i = 0; i < count; ++i)
    {
        GPUParticle p = particles_compact_out[i];
        float proj = dot(push_constants.sort_axis, p.position);
        printf("%f", proj);
    }
    printf("Sorted");
    for (uint i = 0; i < count; ++i)
    {
        GPUParticle p = particles_compact_out[particle_sort[i].index];
        float proj = dot(push_constants.sort_axis, p.position);
        printf("%f", proj);
    }
}

struct VSInput
{
    uint vertex_id: SV_VertexID;
    uint instance_id: SV_InstanceID;
};

struct VSOutput
{
    float4 position: SV_Position;
    [[vk::builtin("PointSize")]] float point_size : PSIZE;
    float3 shadow : COLOR0;
    float4 color : COLOR1;
};

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;

    uint particle_index = particle_sort[input.instance_id].index;
    GPUParticle p = particles[particle_index];
    float4 pos = mul(system_globals.transform, float4(p.position, 1.0));
    float4 view_pos = mul(globals.view, pos);
    output.position = mul(globals.projection, view_pos);

    //printf("wpos vs: %f %f %f", pos);
    const float particle_size = push_constants.particle_size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(globals.projection, corner);
    float point_size = globals.resolution.x * proj_corner.x / proj_corner.w;

    float4 light_view = mul(system_globals.light_view, pos);
    float4 light_clip = mul(system_globals.light_proj, light_view);
    light_clip /= light_clip.w;
    float2 light_uv = light_clip.xy * 0.5 + 0.5;
    light_uv.y = 1.0 - light_uv.y;
    float4 light_sample = light_texture.SampleLevel(light_sampler, light_uv, 0);
    //float3 shadow = 1.0 - light_sample.a * 0.75;
    float3 shadow = 1.0 - light_sample.rgb;

    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    output.shadow = shadow;
    output.color = p.color;
    return output;
}

VSOutput vs_light(VSInput input)
{
    VSOutput output = (VSOutput)0;

    uint particle_index = particle_sort[input.instance_id].index;
    GPUParticle p = particles[particle_index];
    float4 pos = mul(system_globals.transform, float4(p.position, 1.0));
    float4 view_pos = mul(system_globals.light_view, pos);
    output.position = mul(system_globals.light_proj, view_pos);

    const float particle_size = push_constants.particle_size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(system_globals.light_proj, corner);
    // Normally the center in view space would be expected to map to 0, 0 in clip space, 
    // but the shadow map projection has an additional translation that is used to snap 
    // the offset to pixel centers to avoid shimmering under motion.
    float4 center = float4(0, 0, view_pos.z, 1.0);
    float4 proj_center = mul(system_globals.light_proj, center);
    float point_size = system_globals.light_resolution.x * (proj_corner.x - proj_center.x) / proj_corner.w;

    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    output.color = p.color;

    return output;
}

struct PSInput
{
    float4 position: SV_Position;
    float3 shadow : COLOR0;
    float4 color : COLOR1;
};

struct PSOutput
{
    float4 color: SV_Target0;
};

[[vk::ext_builtin_input(/* PointCoord */ 16)]]
static const float2 gl_PointCoord;

PSOutput particle_fs_light(PSInput input)
{
    PSOutput output = (PSOutput)0;

    float dist = distance(gl_PointCoord.xy, float2(0.5, 0.5));
    if (dist > 0.5) discard;

    float alpha = saturate(1.0 - dist * 2.0);
    float4 in_color = push_constants.particle_color * float4(1, 1, 1, alpha);
    
    output.color = float4(in_color.rgb * in_color.a, in_color.a); // Premultiplied alpha

    return output;
}

PSOutput particle_fs_shadowed(PSInput input)
{
    PSOutput output = (PSOutput)0;

    float dist = distance(gl_PointCoord.xy, float2(0.5, 0.5));
    if (dist > 0.5) discard;

    float3 shadow = input.shadow;

    float alpha = saturate(1.0 - dist * 2.0);
    float4 in_color = input.color * float4(1, 1, 1, alpha);
    in_color.rgb *= shadow;
    output.color = float4(in_color.rgb * in_color.a, in_color.a); // Premultiplied alpha

    return output;
}