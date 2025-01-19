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

[[vk::binding(7)]] RWStructuredBuffer<DispatchIndirectCommand> children_emit_indirect_dispatch;

[[vk::push_constant]]
GPUParticlePushConstants push_constants;

static const float3 sphere_center = float3(0, 2, 0);
static const float sphere_radius = 0.5;

float sdf_sphere(float3 p, float3 center, float radius)
{
    return length(p - center) - radius;
}

float sdf(float3 p)
{
    return sdf_sphere(p, sphere_center, sphere_radius);
}

float sdf_func(float3 p)
{
    return (sdf(p));
}

float3 sdf_normal( in float3 p ) // for function f(p)
{
    const float eps = 0.001; // or some other value
    const float2 h = float2(eps,0);
    return normalize( float3(sdf_func(p+h.xyy) - sdf_func(p-h.xyy),
                           sdf_func(p+h.yxy) - sdf_func(p-h.yxy),
                           sdf_func(p+h.yyx) - sdf_func(p-h.yyx) ) );
}


[numthreads(64, 1, 1)]
void emit( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= push_constants.particles_to_spawn)
        return;

    if (particle_system_state[0].active_particle_count >= push_constants.particle_capacity)
        return;

    uint lane_spawn_count = WaveActiveCountBits(true); // == Number of threads that passed the early return checks
    uint local_index = WaveGetLaneIndex();
    uint global_particle_index;

    if (WaveIsFirstLane())
        InterlockedAdd(particle_system_state[0].active_particle_count, lane_spawn_count, global_particle_index);

    global_particle_index = WaveReadLaneFirst(global_particle_index);
    
    uint4 seed = uint4(thread_id.x, globals.frame_index, 69, 720);
    float2 xi = uniform_random(seed).xy;

    GPUParticle p;
    p.velocity = float3(0, 0, 0);
    p.lifetime = 0.2;
    p.position = sphere_center;
    p.color = float4(uniform_random(seed).xyz, 0.1);
    p.size = 0.05;

    p.position += sample_uniform_sphere(xi) * sphere_radius;

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
        float3 n = sdf_normal(p.position);
        float4 phi = gradient_noise_deriv(p.position);
        float3 crossed_grad = cross(n, phi.yzw);
        p.velocity = crossed_grad * 2.0;
        p.position += p.velocity * push_constants.delta_time;
        p.lifetime -= push_constants.delta_time;
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

    indirect_dispatch[0] = command;

    DispatchIndirectCommand child_emit_cmd;
    child_emit_cmd.x = size;
    child_emit_cmd.y = push_constants.children_to_emit;
    child_emit_cmd.z = 1;

    if (particle_system_state[0].active_particle_count > push_constants.particle_capacity)
        printf("WARNING: System trail blazer low on capacity! (%d / %d)", 
            particle_system_state[0].active_particle_count,  push_constants.particle_capacity);

    children_emit_indirect_dispatch[0] = child_emit_cmd;
}

[numthreads(64, 1, 1)]
void write_draw( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= push_constants.num_slices) return;

    uint active_particles = particle_system_state_out[0].active_particle_count;
    uint draw_count = push_constants.num_slices == 0 
        ? active_particles
        : (active_particles + push_constants.num_slices - 1) / push_constants.num_slices;

    DrawIndirectCommand draw_cmd;
    draw_cmd.vertexCount = 1;
    draw_cmd.instanceCount = draw_count;
    draw_cmd.firstVertex = 0;
    draw_cmd.firstInstance = draw_count * thread_id.x;

    indirect_draw[thread_id.x] = draw_cmd;
}

struct VSInput
{
    uint vertex_id: SV_VertexID;
    uint instance_id: SV_InstanceID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float4 color: COLOR0;
    [[vk::builtin("PointSize")]] float point_size : PSIZE;
};

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;

    GPUParticle p = particles[input.instance_id];
    float4 pos = float4(p.position, 1.0);
    float4 view_pos = mul(globals.view, pos);
    output.position = mul(globals.projection, view_pos);

    const float particle_size = p.size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(globals.projection, corner);
    float point_size = globals.resolution.x * proj_corner.x / proj_corner.w;

    output.color = p.color;

    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    return output;
}

struct PSInput
{
    float4 position: SV_Position;
    float4 color: COLOR0;
};

struct PSOutput
{
    float4 color: SV_Target0;
};

[[vk::ext_builtin_input(/* PointCoord */ 16)]]
static const float2 gl_PointCoord;

PSOutput particle_fs(PSInput input)
{
    PSOutput output = (PSOutput)0;

    float dist = distance(gl_PointCoord.xy, float2(0.5, 0.5));
    if (dist > 0.5) discard;

    float3 N;
    N.xy = gl_PointCoord.xy * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float mag2 = dot(N.xy, N.xy);
    N.z = sqrt(1.0 - mag2);
    
    float alpha = saturate(1.0 - dist * 2.0);
    float4 col = input.color;
    col.a *= alpha;
    output.color = float4(col);

    return output;
}