struct VSInput
{
    uint vertex_id: SV_VertexID;
};

struct VSOutput
{
    float4 position: SV_Position;
    float2 texcoord0: TEXCOORD0;
};

[[vk::binding(0)]] SamplerState bilinear_sampler;
[[vk::binding(1)]] Texture2D basecolor_texture;

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float x = float(int(input.vertex_id) - 1);
    float y = float(int(input.vertex_id & 1u) * 2 - 1);
    output.position = float4(x, y, 0.0, 1.0);
    output.texcoord0 = float2(x, y) * 0.5 + 0.5;
    return output;
}

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    float4 basecolor = basecolor_texture.Sample(bilinear_sampler, input.texcoord0);
    output.color = float4(basecolor.rgb, 1.0);

    return output;
}