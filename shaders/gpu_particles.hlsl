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
    p.lifetime = 3.0;
    //p.velocity = point_on_sphere;
    //p.velocity = float3(0, 1, 0);
    p.position = float3(0, 1, 0) + point_on_sphere * push_constants.emitter_radius * sqrt(float(seed.z) / float(0xFFFFFFFFu));
    //p.velocity = curl_noise(p.position);
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

#if 0 
    if (thread_id.x == 0)
    {
        printf("particle count: %d, draw count: %d, slices: %d", 
            active_particles,
            draw_count,
            push_constants.num_slices
        );
    }
#endif

    DrawIndirectCommand draw_cmd;
    draw_cmd.vertexCount = 1;
    draw_cmd.instanceCount = draw_count;
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
        sort.key = sort_key_from_float(asuint(projected)); // TODO: Float sortkey to uint
        
        particle_sort[index] = sort;

#if 0
        if (index == 0)
        {
            AABBPositions aabb;
            aabb.min_x = -1.0;
            aabb.min_y = -1.0;
            aabb.min_z = -1.0;
            aabb.max_x = 1.0;
            aabb.max_y = 1.0;
            aabb.max_z = 1.0;

            aabb_positions[index] = aabb;
        }

        AccelerationStructureInstance instance;
        instance.matrix[0][0] = push_constants.particle_size * 0.5;
        instance.matrix[1][1] = push_constants.particle_size * 0.5;
        instance.matrix[2][2] = push_constants.particle_size * 0.5;
        instance.matrix[0][3] = p.position.x;
        instance.matrix[1][3] = p.position.y;
        instance.matrix[2][3] = p.position.z;
        instance.mask = 0xFF;
        instance.accelerationStructureReference = push_constants.blas_address;
        instances[index] = instance;
#else
        if (index == 0)
        {
            AccelerationStructureInstance instance;
            instance.matrix[0][0] = 1.0;
            instance.matrix[1][1] = 1.0;
            instance.matrix[2][2] = 1.0;
            instance.matrix[0][3] = 0.0;
            instance.matrix[1][3] = 0.0;
            instance.matrix[2][3] = 0.0;
            instance.mask = 0xFF;
            instance.accelerationStructureReference = push_constants.blas_address;
            instances[index] = instance;
        }

        AABBPositions aabb;
        aabb.min_x = p.position.x - push_constants.particle_size * 0.5;
        aabb.min_y = p.position.y - push_constants.particle_size * 0.5;
        aabb.min_z = p.position.z - push_constants.particle_size * 0.5;
        aabb.max_x = p.position.x + push_constants.particle_size * 0.5;
        aabb.max_y = p.position.y + push_constants.particle_size * 0.5;
        aabb.max_z = p.position.z + push_constants.particle_size * 0.5;

        aabb_positions[index] = aabb;
#endif

        
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
    float3 world_pos: POSITION1;
    int out_instance_id: TEXCOORD1;
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

    //printf("wpos vs: %f %f %f", pos);
    const float particle_size = push_constants.particle_size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(globals.projection, corner);
    float point_size = globals.resolution.x * proj_corner.x / proj_corner.w;
    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    output.frag_point_size = output.point_size;
    output.world_pos = pos.xyz;
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
    output.center_pos = output.position.xy / output.position.w;

    const float particle_size = push_constants.particle_size;
    float4 corner = float4(particle_size * 0.5, particle_size * 0.5, view_pos.z, 1.0);
    float4 proj_corner = mul(system_globals.light_proj, corner);
    float point_size = system_globals.light_resolution.x * proj_corner.x / proj_corner.w;

    output.point_size = p.lifetime > 0.0 ? max(point_size, 0.71) : 0.0f;
    output.frag_point_size = output.point_size;
    output.world_pos = pos.xyz;
    output.out_instance_id = input.instance_id;

    return output;
}

struct PSInput
{
    float4 position: SV_Position;
    float2 center_pos: POSITION0;
    float3 world_pos: POSITION1;
    int out_instance_id: TEXCOORD1;
    float frag_point_size : TEXCOORD0;
};

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput particle_fs_light(PSInput input)
{
    PSOutput output = (PSOutput)0;

    float2 center_pos = (input.center_pos * 0.5 + 0.5) * system_globals.light_resolution;
    //center_pos.y = system_globals.light_resolution.y - center_pos.y;
#if 0
    if (input.out_instance_id == 0)
        printf("lr: %u %u, input pos: %f %f, in_center: %f %f, center_pos: %f %f, point size: %f", 
            system_globals.light_resolution.x, system_globals.light_resolution.y,
            input.position.x, input.position.y, 
            input.center_pos.x, input.center_pos.y,
            center_pos.x, center_pos.y, input.frag_point_size);
#endif
    float r = distance(center_pos, input.position.xy);
    r /= (input.frag_point_size * 0.5);
    //float alpha = 1.0 - smoothstep(0.0, 1.0, r);
    //float alpha = 1.0 - saturate(r);
    float alpha = r < 1.0 ? 1.0 : 0.0;

    float4 in_color = push_constants.particle_color * float4(1, 1, 1, alpha);
    output.color = float4(in_color.rgb * in_color.a, in_color.a); // Premultiplied alpha

    return output;
}

PSOutput particle_fs_shadowed(PSInput input)
{
    PSOutput output = (PSOutput)0;

    float2 uv = input.position.xy / globals.resolution;
    float2 clip_pos = uv * 2.0 - 1.0;
    clip_pos.y = -clip_pos.y;

    float2 center_pos = (input.center_pos * 0.5 + 0.5) * globals.resolution;
    center_pos.y = globals.resolution.y - center_pos.y;
    float r = distance(center_pos, input.position.xy);
    r /= (input.frag_point_size * 0.5);
    //float alpha = 1.0 - smoothstep(0.0, 1.0, r);
    float alpha = 1.0 - saturate(r);
    //float alpha = r < 1.0 ? 1.0 : 0.0;

    float4 view_pos = mul(globals.projection_inverse, float4(clip_pos.xy, input.position.z, 1.0));
    float4 world_pos = mul(globals.view_inverse, view_pos);
    world_pos /= world_pos.w;

    float4 light_view = mul(system_globals.light_view, world_pos);
    float4 light_clip = mul(system_globals.light_proj, light_view);
    light_clip /= light_clip.w;

    float2 light_uv = light_clip * 0.5 + 0.5;
    //light_uv.y = 1.0 - light_uv.y;
    // Flip?

    float4 light_sample = light_texture.Sample(light_sampler, light_uv);
    float3 shadow = 1.0 - light_sample.rgb;

    float3 fake_normal_ss = float3(0, 0, 1);
    float3 edge_normal_ss = normalize(float3(input.position.xy - center_pos.xy, 0.0));
    if (length(edge_normal_ss) > 1e-2)
    {
        fake_normal_ss = normalize(lerp(fake_normal_ss, edge_normal_ss, r));
    }

    fake_normal_ss.y = -fake_normal_ss.y;

    float3 fake_normal_ws = normalize(mul(globals.view_inverse, float4(fake_normal_ss, 0.0)).xyz);
    float NoL = saturate(dot(fake_normal_ws, globals.sun_direction.xyz));

    float4 in_color = push_constants.particle_color * float4(1, 1, 1, alpha);
    in_color.rgb *= shadow;
    //in_color.rgb += push_constants.particle_color * 0.1;
    //in_color.rgb = fake_normal_ss * 0.5 + 0.5;
    //in_color.rgb = NoL;
    output.color = float4(in_color.rgb * in_color.a, in_color.a); // Premultiplied alpha

    return output;
}