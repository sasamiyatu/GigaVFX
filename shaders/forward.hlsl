struct VSInput
{
    uint vertex_id: SV_VertexID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float2 texcoord0: TEXCOORD0;
    float3 normal: NORMAL;
};

[[vk::binding(0, 1)]] Texture2D<float4> bindless_textures[];

[[vk::binding(0)]] SamplerState bilinear_sampler;
[[vk::binding(1)]] cbuffer globals {
    float4x4 view;
    float4x4 projection;
    float4x4 viewprojection;
}

struct PushConstants
{
    float4x4 model;
    uint64_t position_buffer;
    uint64_t normal_buffer;
    uint64_t tangent_buffer;
    uint64_t texcoord0_buffer;
    uint64_t texcoord1_buffer;
    int basecolor_texture;
};

[[vk::push_constant]]
PushConstants push_constants;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float3 pos = vk::RawBufferLoad<float3>(push_constants.position_buffer + input.vertex_id * 12);
    float3 world_pos = mul(push_constants.model, float4(pos, 1.0)).xyz;
    output.position = mul(viewprojection, float4(world_pos, 1.0));
    if (push_constants.normal_buffer)
        output.normal = vk::RawBufferLoad<float3>(push_constants.normal_buffer + input.vertex_id * 12);
    if (push_constants.texcoord0_buffer)
        output.texcoord0 = vk::RawBufferLoad<float2>(push_constants.texcoord0_buffer + input.vertex_id * 8);
    return output;
}

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    //float4 basecolor = basecolor_texture.Sample(bilinear_sampler, input.texcoord0);
    float4 basecolor = bindless_textures[push_constants.basecolor_texture].Sample(bilinear_sampler, input.texcoord0);
    float3 N = normalize(input.normal);
    output.color = float4(N * 0.5 + 0.5, 1.0);
    output.color = basecolor;

    return output;
}