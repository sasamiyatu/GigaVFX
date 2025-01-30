#include "shared.h"

[[vk::binding(0)]] StructuredBuffer<GPUParticleSystemState> system_states;
[[vk::binding(1)]] RWStructuredBuffer<DrawIndirectCommand> indirect_draw;

struct PushConstants
{
    uint system_count;
};

[[vk::push_constant]]
PushConstants push_constants;

[numthreads(64, 1, 1)]
void write_draw( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= push_constants.system_count) return;

    DrawIndirectCommand command;
    command.vertexCount = 1;
    command.instanceCount = system_states[thread_id.x].active_particle_count;
    command.firstVertex = 0;
    command.firstInstance = 0;

    indirect_draw[thread_id.x] = command;
}

