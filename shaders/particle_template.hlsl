#include "shared.h"
// #include "math.hlsli"
// #include "random.hlsli"
// #include "noise.hlsli"
// #include "misc.hlsli"

// The following functions are expected to be defined in a file that is included at compile time:
bool particle_init(inout GPUParticle p, float delta_time, uint4 seed);
bool particle_update(inout GPUParticle p, float delta_time, uint4 seed);

[[vk::binding(0)]] cbuffer globals {
    ShaderGlobals globals;
}

[[vk::binding(1)]] RWStructuredBuffer<GPUParticle> particles;
[[vk::binding(2)]] RWStructuredBuffer<GPUParticleSystemState> particle_system_state;

[[vk::binding(3)]] RWStructuredBuffer<GPUParticle> particles_compact_out;
[[vk::binding(4)]] RWStructuredBuffer<GPUParticleSystemState> particle_system_state_out;

[[vk::binding(5)]] RWStructuredBuffer<DispatchIndirectCommand> indirect_dispatch;
[[vk::binding(6)]] RWStructuredBuffer<DrawIndirectCommand> indirect_draw;

[[vk::push_constant]]
ParticleTemplatePushConstants push_constants;

[numthreads(64, 1, 1)]
void emit( uint3 thread_id : SV_DispatchThreadID )
{
    // if (thread_id.x >= push_constants.particles_to_spawn)
    //     return;

    if (thread_id.x == 0)
    {
        indirect_dispatch[0].y = 1;
        indirect_dispatch[0].z = 1;
    }

    if (particle_system_state[0].active_particle_count >= push_constants.particle_capacity)
        return;

    uint4 seed = uint4(thread_id.xy, globals.frame_index, 42);
    GPUParticle p = (GPUParticle)0;
    bool spawned = particle_init(p, push_constants.delta_time, seed);
    if (!spawned)
        return;

    uint lane_spawn_count = WaveActiveCountBits(true); // == Number of threads that passed the early return checks
    uint local_index = WaveGetLaneIndex();
    uint global_particle_index;

    if (WaveIsFirstLane())
    {
        InterlockedAdd(particle_system_state[0].active_particle_count, lane_spawn_count, global_particle_index);
        uint size = (lane_spawn_count + global_particle_index + 63) / 64;
        InterlockedMax(indirect_dispatch[0].x, size);
    }

    global_particle_index = WaveReadLaneFirst(global_particle_index);

    particles[global_particle_index + local_index] = p;
}

[numthreads(64, 1, 1)]
void simulate( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    if (thread_id.x == 0)
    {
        indirect_draw[0].vertexCount = 1;
        indirect_draw[0].firstVertex = 0;
        indirect_draw[0].firstInstance = 0;
    }

    GPUParticle p = particles[thread_id.x];
    uint4 seed = uint4(thread_id.xy, globals.frame_index, 1337);
    bool alive = particle_update(p, push_constants.delta_time, seed);

    uint local_index = WavePrefixCountBits(alive);
    uint alive_count = WaveActiveCountBits(alive);

    if (alive_count == 0) return;

    uint global_particle_index;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(particle_system_state_out[0].active_particle_count, alive_count, global_particle_index);
        InterlockedAdd(indirect_draw[0].instanceCount, alive_count);
    }

    global_particle_index = WaveReadLaneFirst(global_particle_index);

    if (alive)
    {
        uint index = global_particle_index + local_index;
        particles_compact_out[index] = p;
    }
}