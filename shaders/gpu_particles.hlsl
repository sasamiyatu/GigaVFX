#include "shared.h"
#include "math.hlsli"
#include "random.hlsli"
#include "noise.hlsli"
#include "misc.hlsli"

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

[[vk::binding(6)]] RWStructuredBuffer<GPUParticleIndirectData> indirect_dispatch;
[[vk::binding(7)]] RWStructuredBuffer<GPUParticleSort> particle_sort;

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
    p.lifetime = 3.0;
    //p.velocity = point_on_sphere;
    p.position = float3(0, 1, 0) + point_on_sphere * 0.1;
    p.velocity = curl_noise(p.position);
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
        p.velocity = curl_noise(p.position);
        uint4 seed = uint4(globals.frame_index, asuint(p.position));
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

    DrawIndirectCommand draw_cmd;
    draw_cmd.vertexCount = 1;
    draw_cmd.instanceCount = particle_system_state[0].active_particle_count;
    draw_cmd.firstVertex = 0;
    draw_cmd.firstInstance = 0;

    indirect_dispatch[0].dispatch_cmd = command;
    indirect_dispatch[0].draw_cmd = draw_cmd;
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
        sort.key = sort_key_from_float(asuint(projected)); // TODO: Float sortkey to uint
        
        particle_sort[index] = sort;
    }
}

// Used for debugging only
[numthreads(1, 1, 1)]
void cs_debug_print_sorted_particles( uint3 thread_id : SV_DispatchThreadID )
{
    uint count = particle_system_state_out[0].active_particle_count;
    printf("active count %d, sort_axis: (%f, %f, %f)\n", count, push_constants.sort_axis);

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
    float2 center_pos: POSITION0;
    [[vk::builtin("PointSize")]] float point_size : PSIZE;
    float frag_point_size : TEXCOORD0;
};

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;

    uint particle_index = particle_sort[input.instance_id].index;
    GPUParticle p = particles[particle_index];
    float4 pos = mul(system_globals.transform, float4(p.position, 1.0));
    float4 view_pos = mul(globals.view, pos);
    output.position = mul(globals.projection, view_pos);
    output.center_pos = output.position.xy / output.position.w;

    const float particle_size = push_constants.particle_size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(globals.projection, corner);
    float point_size = globals.resolution.x * proj_corner.x / proj_corner.w;
    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    output.frag_point_size = output.point_size;
    return output;
}

struct PSInput
{
    float4 position: SV_Position;
    float2 center_pos: POSITION0;
    float frag_point_size : TEXCOORD0;
};

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(PSInput input)
{
    PSOutput output = (PSOutput)0;

    float2 uv = input.position.xy / globals.resolution;
    float2 clip_pos = uv * 2.0 - 1.0;
    clip_pos.y = -clip_pos.y;

    float2 center_pos = (input.center_pos * 0.5 + 0.5) * globals.resolution;
    center_pos.y = globals.resolution.y - center_pos.y;
    float r = distance(center_pos, input.position.xy);
    r /= (input.frag_point_size * 0.5);
    float alpha = 1.0 - smoothstep(0.0, 1.0, r);

    float4 in_color = push_constants.particle_color;
    output.color = in_color * float4(1, 1, 1, alpha);

    return output;
}