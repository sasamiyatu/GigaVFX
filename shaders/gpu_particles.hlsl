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
[[vk::binding(14)]] SamplerState sdf_sampler;
[[vk::binding(15)]] Texture3D sdf_texture;

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
    p.position = float3(0, 1, 0) + point_on_sphere * push_constants.emitter_radius * sqrt(float(seed.z) / float(0xFFFFFFFFu));
    particles[global_particle_index + local_index] = p;
}

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

    float d = (sdf_texture.SampleLevel(sdf_sampler, grid_uv, 0)).x;
    return d;
}

float sdf_func(float3 p)
{
    return abs(sdf(p));
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
void emit_sphere( uint3 thread_id : SV_DispatchThreadID )
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
    float3 grid_size = float3(1, 1, 1);
    float3 point_on_grid = uniform_random(seed).xyz * grid_size;
    float3 point_on_sphere = sample_uniform_sphere(uniform_random(seed).xy);
    GPUParticle p;
    p.velocity = 0;
    p.lifetime = 100.0;
    
    const float sdf_scale = 0.1;
    float3 bb_min = push_constants.sdf_origin * sdf_scale;
    float3 bb_max = (push_constants.sdf_origin + push_constants.sdf_grid_spacing * float3(push_constants.sdf_grid_dims)) * sdf_scale;
    float3 extent = bb_max - bb_min;

    p.position = float3(bb_min.x + 0.1, bb_min.y + 0.1, bb_min.z + 0.1) + uniform_random(seed).xyz * float3(extent.x - 0.2, 0, extent.z - 0.2);
    // float3 pos;
    // do {
    //     pos = bb_min + extent * uniform_random(seed).xyz;
    // } while (sdf(pos) > 0);
    // p.position = pos;
    particles[global_particle_index + local_index] = p;
}

[numthreads(64, 1, 1)]
void update_simple( uint3 thread_id : SV_DispatchThreadID )
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


        float damping = pow(smoothstep(0, 1, d) * 0.9 + 0.1, push_constants.delta_time * 0.2);
        //p.velocity = sdf_normal(p.position) * 0.1 + curl_noise(p.position, 0) * 0.1;
        float3 attraction = -sdf_gradient(p.position);

        //if (thread_id.x == 0) printf("d: %f -grad: (%f  %f %f)", d, attraction.x, attraction.y, attraction.z);
        float3 c = curl_noise(p.position, 0);
        p.velocity += (attraction + curl * 0.1)* push_constants.delta_time * 0.2;
        //p.velocity += (curl * 0.1) * push_constants.delta_time;
        //p.velocity += (attraction) * push_constants.delta_time;
        //p.velocity += (curl * 0.1 + to_surface)  * push_constants.delta_time;
        p.velocity *= damping;
        p.position += p.velocity * push_constants.delta_time * 0.2;
        p.lifetime -= push_constants.delta_time;
        //if (sdf_func(p.position) > 0.01) p.lifetime = -1.0;
    }

    particles[thread_id.x] = p;
}

[numthreads(64, 1, 1)]
void cs_simulate_particles( uint3 thread_id : SV_DispatchThreadID )
{
    if (thread_id.x >= particle_system_state[0].active_particle_count)
        return;

    GPUParticle p = particles[thread_id.x];
    if (p.lifetime > 0.0)
    {
        p.velocity = (float3(1, 0, 0) * 3.0 + curl_noise(p.position, push_constants.time * push_constants.noise_time_scale) * push_constants.noise_scale) * push_constants.speed;
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

    return output;
}

struct PSInput
{
    float4 position: SV_Position;
    float3 shadow : COLOR0;
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
    float4 in_color = push_constants.particle_color * float4(1, 1, 1, alpha);
    in_color.rgb *= shadow;
    output.color = float4(in_color.rgb * in_color.a, in_color.a); // Premultiplied alpha

    return output;
}