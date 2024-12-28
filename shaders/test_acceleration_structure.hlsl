#include "shared.h"

[[vk::binding(0)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(1)]] cbuffer globals {
    ShaderGlobals globals;
}
[[vk::binding(2)]] RWTexture2D<float4> out_texture;
[[vk::binding(3)]] StructuredBuffer<AABBPositions> aabb_positions;

float2 sphere_intersect(in float3 ro, in float3 rd, float3 center, float radius)
{
    float3 oc = ro - center;
    
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, oc);
    float c = dot(oc, oc) - radius * radius;
    
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) return float2(-1, -1);
    
    float d = sqrt(discriminant);
    
    float a2 = 2.0 * a;
    
    return float2((-b - d) / a2, (-b + d) / a2);
}

float3 create_camera_ray(float2 uv, float4x4 proj, float4x4 v_inv)
{
    float aspect = proj[1][1] / proj[0][0];
    float tan_half_fov_y = 1.f / proj[1][1];
    float3 rd = normalize(
        (uv.x * v_inv[0].xyz * tan_half_fov_y * aspect) - 
        (uv.y * v_inv[1].xyz * tan_half_fov_y) -
        v_inv[2].xyz);
    return rd;
}

[numthreads(8, 8, 1)]
void test_acceleration_structure( uint3 thread_id : SV_DispatchThreadID )
{
    RayDesc ray;

    if (thread_id.x >= globals.resolution.x || thread_id.y >= globals.resolution.y) return;
    float2 uv = float2(thread_id.xy) / float2(globals.resolution);

    uv = uv * 2.0 - 1.0;
    float3 rd = create_camera_ray(uv, transpose(globals.projection), transpose(globals.view_inverse));
    float3 ro = transpose(globals.view_inverse)[3].xyz;

    ray.Origin = ro;
    ray.TMin = 0.0;
    ray.Direction = rd;
    ray.TMax = 1e38f;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(acceleration_structure, 0, 0xFFFFFFFF, ray);

    float3 closest_center = 0;
    while(q.Proceed())
    {
        switch(q.CandidateType())
        {
        case CANDIDATE_PROCEDURAL_PRIMITIVE:
        {
            uint pi = q.CandidatePrimitiveIndex();

            AABBPositions aabb = aabb_positions[pi];
            float3 center = float3(aabb.min_x + aabb.max_x, aabb.min_y + aabb.max_y, aabb.min_z + aabb.max_z) * 0.5;
            float radius = (aabb.max_x - aabb.min_x) * 0.5;
            float3 ro = q.CandidateObjectRayOrigin();
            float3 rd = q.CandidateObjectRayDirection();
            float2 shit = sphere_intersect(ro, rd, center, radius);
            float t = shit.x > 0.0 ? shit.x : shit.y;
            //float t = 1.0f;
            if ((q.RayTMin() <= t) && (t <= q.CommittedRayT()))
            {
                q.CommitProceduralPrimitiveHit(t);
                closest_center = center;
            }
            break;
        }
        default:
            break;
        }
    }

    if (q.CommittedStatus() != COMMITTED_NOTHING)
    {
        float t = q.CommittedRayT();
        float3 p = ro + rd * t;
        float3 n = normalize(p - closest_center);
        out_texture[thread_id.xy] = float4(n * 0.5 + 0.5, 1);

    }
}