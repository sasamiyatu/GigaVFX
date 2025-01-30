#include "shared.h"

[[vk::binding(0)]] StructuredBuffer<GPUParticleSystemState> system_states;

[[vk::binding(1)]] RWStructuredBuffer<DispatchIndirectCommand> indirect_dispatch;

struct PushConstants
{
    uint system_count;
};

[[vk::push_constant]]
PushConstants push_constants;

[numthreads(64, 1, 1)]
void write_dispatch( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= push_constants.system_count) return;

    uint size = (system_states[thread_id.x].active_particle_count + 63) / 64;

    DispatchIndirectCommand command;
    command.x = size;
    command.y = 1;
    command.z = 1;

    indirect_dispatch[thread_id.x] = command;
}

