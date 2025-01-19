#include "shared.h"
#include "math.hlsli"
#include "random.hlsli"
#include "noise.hlsli"
#include "misc.hlsli"

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] RWStructuredBuffer<GPUParticle> particles;
[[vk::binding(2)]] RWStructuredBuffer<GPUParticleSystemState> particle_system_state;

[[vk::binding(3)]] RWStructuredBuffer<GPUParticle> particles_compact_out;
[[vk::binding(4)]] RWStructuredBuffer<GPUParticleSystemState> particle_system_state_out;

[[vk::binding(5)]] RWStructuredBuffer<DispatchIndirectCommand> indirect_dispatch;
[[vk::binding(6)]] RWStructuredBuffer<DrawIndirectCommand> indirect_draw;

[[vk::binding(7)]] StructuredBuffer<GPUParticleSystemState> parent_system_state;
[[vk::binding(8)]] StructuredBuffer<GPUParticle> parent_particles;

[[vk::push_constant]]
TrailBlazerChildPushConstants push_constants;

static const float lifetime = 0.2f;

[numthreads(64, 1, 1)]
void emit( uint3 thread_id : SV_DispatchThreadID )
{
    uint parent_count = parent_system_state[0].active_particle_count;
    if (thread_id.x >= parent_count)
        return;

    if (particle_system_state[0].active_particle_count >= push_constants.particle_capacity)
        return;

    uint lane_spawn_count = WaveActiveCountBits(true); // == Number of threads that passed the early return checks
    uint local_index = WaveGetLaneIndex();
    uint global_particle_index;

    if (WaveIsFirstLane())
        InterlockedAdd(particle_system_state[0].active_particle_count, lane_spawn_count, global_particle_index);

    global_particle_index = WaveReadLaneFirst(global_particle_index);

    GPUParticle parent = parent_particles[thread_id.x];

    GPUParticle p;
    p.velocity = 0;
    p.lifetime = lifetime;
    p.size = 0.007f;
    p.color = float4(0.3, 0.7, 0.3, 0.2);
    p.color = parent.color;
    p.position = parent.position;

    uint4 seed = uint4(thread_id.x, thread_id.y, globals.frame_index, 42);
    float4 xi = uniform_random(seed);

    float3 pos_in_sphere = sample_uniform_sphere(xi.xy);
    //p.position += pos_in_sphere * 0.005f;
    p.position += parent.velocity * (2.0 * xi.z - 1.0) * push_constants.delta_time;
    p.velocity = pos_in_sphere * 0.005f;

    particles[global_particle_index + local_index] = p;
}

[numthreads(64, 1, 1)]
void simulate( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    GPUParticle p = particles[thread_id.x];
    if (p.lifetime > 0.0)
    {
        p.velocity += (0, -9.8, 0) * push_constants.delta_time;
        p.size = (p.lifetime / lifetime) * 0.01f;
        p.position += p.velocity * push_constants.delta_time;
        p.lifetime -= push_constants.delta_time;
        p.color.a = saturate(p.lifetime / lifetime);
    }

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
    }
}

[numthreads(1, 1, 1)]
void write_dispatch( uint3 thread_id : SV_DispatchThreadID )
{
    uint size = (particle_system_state[0].active_particle_count + 63) / 64;

    DispatchIndirectCommand command;
    command.x = size;
    command.y = 1;
    command.z = 1;

    if (particle_system_state[0].active_particle_count > push_constants.particle_capacity)
        printf("WARNING: System trail blazer child low on capacity! (%d / %d)", 
            particle_system_state[0].active_particle_count,  push_constants.particle_capacity);

    indirect_dispatch[0] = command;
}

[numthreads(1, 1, 1)]
void write_draw( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= 1) return;

    uint active_particles = particle_system_state_out[0].active_particle_count;
    uint draw_count = active_particles;

    DrawIndirectCommand draw_cmd;
    draw_cmd.vertexCount = 1;
    draw_cmd.instanceCount = draw_count;
    draw_cmd.firstVertex = 0;
    draw_cmd.firstInstance = 0;

    indirect_draw[thread_id.x] = draw_cmd;
}
