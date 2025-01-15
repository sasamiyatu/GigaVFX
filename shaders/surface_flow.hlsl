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
[[vk::binding(7)]] SamplerState sdf_sampler;
[[vk::binding(8)]] Texture3D sdf_texture;

[[vk::binding(9)]] RWStructuredBuffer<uint> grid_counters;
[[vk::binding(10)]] RWStructuredBuffer<uint> grid_cells;

[[vk::push_constant]]
GPUParticlePushConstants push_constants;

float sdTorus( float3 p, float2 t )
{
    float2 q = float2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float sdf_sphere(float3 p, float3 center, float radius)
{
    return length(p - center) - radius;
}

float sdBoxFrame( float3 p, float3 b, float e )
{
    p = abs(p  )-b;
    float3 q = abs(p+e)-e;
    return min(min(
        length(max(float3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
        length(max(float3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
        length(max(float3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

float sdBox( float3 p, float3 b )
{
  float3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float smax(float a, float b, float k)
{
    return log(exp(k*a)+exp(k*b))/k;
}

float4 get_texel( float3 p, float3 dims )
{
    p = p * dims + 0.5;

    float3 i = floor(p);
    float3 f = p - i;
    f = f*f*f*(f*(f*6.0-15.0)+10.0);
    p = i + f;

    p = (p - 0.5)/ dims;
    return sdf_texture.SampleLevel(sdf_sampler, p, 0);
}

float sdf(float3 p)
{
    //return abs(sdf_sphere(p, float3(0, 2, 0), 0.5));
    //return abs(sdTorus(float3(p.x, p.y - 1.0, p.z), float2(0.5, 0.15)));
    //return abs(sdBoxFrame(float3(p.x, p.y - 2.0, p.z), float3(0.5,0.3,0.5), 0.025));

    const float sdf_scale = 0.1;
    float3 bb_min = push_constants.sdf_origin * sdf_scale;
    float3 bb_max = (push_constants.sdf_origin + push_constants.sdf_grid_spacing * float3(push_constants.sdf_grid_dims)) * sdf_scale;
    float3 extent = bb_max - bb_min;
    float3 local_pos = p - bb_min;
    float3 grid_uv = local_pos / extent;

    //return sdf_sphere(p, bb_min + extent * 0.5, 0.5 * min(extent.z, min(extent.x, extent.y)));

    float d = (sdf_texture.SampleLevel(sdf_sampler, grid_uv, 0)).x * sdf_scale;
    return d;
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

float3 sdf_gradient(in float3 p)
{
    const float eps = 0.001; // or some other value
    const float2 h = float2(eps,0);
    return float3(sdf_func(p+h.xyy) - sdf_func(p-h.xyy),
                           sdf_func(p+h.yxy) - sdf_func(p-h.yxy),
                           sdf_func(p+h.yyx) - sdf_func(p-h.yyx) ) / (2.0 * h.x);
}

float3 scaled_normal(float3 p)
{
    float3 normal = sdf_normal(p);
    float offset = smoothstep(-0.7, 0.7, gradient_noise3d(p));
    return normal * offset;
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

    uint4 seed = uint4(thread_id.x, globals.frame_index, 42, 1337);
    float3 grid_size = float3(1, 1, 1);
    float3 point_on_grid = uniform_random(seed).xyz * grid_size;
    float3 point_on_sphere = sample_uniform_sphere(uniform_random(seed).xy);
    GPUParticle p;
    p.velocity = 0;
    p.lifetime = 10.0;
    
    const float sdf_scale = 0.1;
    float3 bb_min = push_constants.sdf_origin * sdf_scale;
    float3 bb_max = (push_constants.sdf_origin + push_constants.sdf_grid_spacing * float3(push_constants.sdf_grid_dims)) * sdf_scale;
    float3 extent = bb_max - bb_min;

    float3 pos = bb_min + extent * (0.1 + 0.8 * uniform_random(seed).xyz);
    float3 n = sdf_normal(pos);
    float d = sdf_func(pos);
    pos -= n * d;
    p.position = pos;
    //p.position = float3(bb_min.x + 0.1, bb_min.y + 0.1, bb_min.z + 0.1) + uniform_random(seed).xyz * float3(extent.x - 0.2, 0, extent.z - 0.2);
    //uint x = thread_id.x % push_constants.sdf_grid_dims.x;
    //uint z = thread_id.x / push_constants.sdf_grid_dims.x;
    //p.position = push_constants.sdf_origin + push_constants.sdf_grid_spacing * float3(x + 0.5, 0, z + 0.5) + float3(0, 1, 0);
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
        float2 h = float2(1e-3f, 0);
        float span = h.x * 2.0;
        float3 dx = (scaled_normal(p.position + h.xyy) - scaled_normal(p.position - h.xyy)) / span;
        float3 dy = (scaled_normal(p.position + h.yxy) - scaled_normal(p.position - h.yxy)) / span;
        float3 dz = (scaled_normal(p.position + h.yyx) - scaled_normal(p.position - h.yyx)) / span;

        float3 curl = float3(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x);
        float3 n = sdf_normal(p.position);
        float3 to_surface = sdf_func(p.position) > 0.0 ? -n : n;
        float d = sdf_func(p.position);
        float3 gradient = sdf_gradient(p.position);

        //float damping = pow(smoothstep(0, 1, d) * 0.9 + 0.1, push_constants.delta_time);
        float3 attraction = -sdf_gradient(p.position) * 3.0;

        float3 c = float3(
            gradient_noise3d(p.position), 
            gradient_noise3d(p.position + float3(123.4, 5123.1, -1230.5)), 
            gradient_noise3d(p.position + float3(9812.2, 70124.2, 75912.5))
        );

        float4 phi = gradient_noise_deriv(p.position);
        float4 psi = gradient_noise_deriv((p.position +  float3(31.416, -47.853, 12.679)));
        float3 crossed_grad = cross(n, phi.yzw);
        //float dotted = dot(normalize(phi), normalize(psi));
        // if (all(abs(crossed_grad) < 1e-3))
        // {
        //     printf("Zero velocity (%f %f %f), phi: (%f %f %f), psi: (%f %f %f) d: %f", crossed_grad.x, crossed_grad.y, crossed_grad.z, phi.y, phi.z, phi.w, psi.y, psi.z, psi.w, dotted);
        // }
        p.velocity = crossed_grad;
        //p.velocity += (attraction + curl * 0.5) * push_constants.delta_time;
        //if (d < 0.01) p.velocity -= dot(gradient, p.velocity) * gradient;
        //p.velocity *= 0.99;
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

    DrawIndirectCommand draw_cmd;
    draw_cmd.vertexCount = 1;
    draw_cmd.instanceCount = draw_count;
    draw_cmd.firstVertex = 0;
    draw_cmd.firstInstance = draw_count * thread_id.x;

    indirect_draw[thread_id.x] = draw_cmd;
}

// TODO: Should probably be merged with simulate
[numthreads(64, 1, 1)]
void compact( uint3 thread_id : SV_DispatchThreadID )
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
    }
}

[numthreads(64, 1, 1)]
void update_grid( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count) 
        return;

    GPUParticle p = particles[thread_id.x];

    uint3 dims = push_constants.sdf_grid_dims;
    uint3 cell = floor((p.position - push_constants.sdf_origin) / push_constants.sdf_grid_spacing);
    cell = clamp(cell, uint3(0, 0, 0), dims - 1);
    uint linear_idx = cell.x + cell.y * dims.x + cell.z * dims.x * dims.y;
    uint index_in_cell;
    InterlockedAdd(grid_counters[linear_idx], 1, index_in_cell);
    if (index_in_cell < push_constants.max_particles_in_cell)
    {
        grid_cells[linear_idx * push_constants.max_particles_in_cell + index_in_cell] = thread_id.x;
    }
    else
    {
        //printf("cell (%d %d %d) has too many particles (%d), ac: %d!", cell.x, cell.y, cell.z, index_in_cell, particle_system_state[0].active_particle_count);
    }
}

float3 collide_spheres(float3 pa, float3 pb, float3 va, float3 vb)
{
    const float spring = 0.5;
    const float damping = 0.02;
    const float shear = 0.1;

    float3 relative_pos = pb - pa;
    float dist = length(relative_pos);
    float collide_dist = push_constants.particle_size;

    float3 force = 0;

    if (dist < collide_dist) 
    {
        float3 norm = relative_pos / dist;

        // relative velocity
        float3 relative_vel = vb - va;

        // relative tangential velocity
        float3 tan_vel = relative_vel - (dot(relative_vel, norm) * norm);

        // spring force
        force = -spring * (collide_dist - dist) * norm;
        // dashpot (damping) force
        force += damping * relative_vel;
        // tangential shear force
        force += shear * tan_vel;
        // // attraction
        // force += attraction * relPos;
    } 

    return float3(0, 0, 0);

    return force;
}

[numthreads(64, 1, 1)]
void resolve_collisions( uint3 thread_id : SV_DispatchThreadID)
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    if (thread_id.x == 0)
    {
        particle_system_state_out[0].active_particle_count = particle_system_state[0].active_particle_count;
    }
    
    GPUParticle p = particles[thread_id.x];

    int3 p_cell = floor((p.position - push_constants.sdf_origin) / push_constants.sdf_grid_spacing);
    uint3 dims = push_constants.sdf_grid_dims;

    float3 force = 0;
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int3 cell = p_cell + int3(x, y, z);
                if (any(cell < 0) || any(cell >= push_constants.sdf_grid_dims))
                    continue;

                uint linear_idx = cell.x + cell.y * dims.x + cell.z * dims.x * dims.y;
                for (uint i = 0; i < grid_counters[linear_idx]; ++i)
                {
                    uint pi = grid_cells[linear_idx * push_constants.max_particles_in_cell + i];
                    if (pi != thread_id.x)
                    {
                        GPUParticle np = particles[pi];
                        force += collide_spheres(p.position, np.position, p.velocity, np.velocity);
                    }
                }
            }
        }
    }

    if (thread_id.x == 0) printf("force: (%f %f %f)", force.x, force.y, force.z);
    p.velocity += force;
    particles_compact_out[thread_id.x] = p;
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
    [[vk::builtin("PointSize")]] float point_size : PSIZE;
};

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;

    GPUParticle p = particles[input.instance_id];
    float4 pos = float4(p.position, 1.0);
    float4 view_pos = mul(globals.view, pos);
    output.position = mul(globals.projection, view_pos);

    const float particle_size = push_constants.particle_size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(globals.projection, corner);
    float point_size = globals.resolution.x * proj_corner.x / proj_corner.w;

    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    return output;
}

struct PSInput
{
    float4 position: SV_Position;
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
    float4 in_color = push_constants.particle_color * float4(1, 1, 1, alpha);
    output.color = float4(N * 0.5 + 0.5, 1 * 0.99 + 0.01 * push_constants.particle_color.a);

    return output;
}