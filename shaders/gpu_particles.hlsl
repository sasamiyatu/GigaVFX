#include "shared.h"
#include "math.hlsli"
#include "random.hlsli"

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

[[vk::push_constant]]
GPUParticlePushConstants push_constants;

[numthreads(64, 1, 1)]
void cs_emit_particles( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= push_constants.particles_to_spawn)
        return;

    if (particle_system_state[0].active_particle_count >= system_globals.particle_capacity)
        return;

    uint particle_index;
    InterlockedAdd(particle_system_state[0].active_particle_count, 1, particle_index);

    uint4 seed = uint4(thread_id.x, globals.frame_index, 42, 1337);
    float3 point_on_sphere = sample_uniform_sphere(uniform_random(seed).xy);
    GPUParticle p;
    p.lifetime = 3.0;
    p.velocity = point_on_sphere;
    p.position = float3(0, 1, 0) + point_on_sphere * 0.1;
    particles[particle_index] = p;
}

[numthreads(64, 1, 1)]
void cs_simulate_particles( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    GPUParticle p = particles[thread_id.x];
    if (p.lifetime > 0.0)
    {
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
void cs_compact_particles( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    GPUParticle p = particles[thread_id.x];
    if (p.lifetime > 0.0)
    {
        uint particle_index;
        InterlockedAdd(particle_system_state_out[0].active_particle_count, 1, particle_index);
        particles_compact_out[particle_index] = p;
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

    GPUParticle p = particles[input.instance_id];
    float4 pos = mul(system_globals.transform, float4(p.position, 1.0));
    float4 view_pos = mul(globals.view, pos);
    output.position = mul(globals.projection, view_pos);
    output.center_pos = output.position.xy / output.position.w;

    const float particle_size = 0.1f;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(globals.projection, corner);
    float point_size = globals.resolution.x * proj_corner.x / proj_corner.w;
    output.point_size = p.lifetime > 0.0 ? point_size : 0.0f;
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

    output.color = float4(1.0, 0.6, 0.4, alpha);

    return output;
}