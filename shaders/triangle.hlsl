struct VSInput
{
    uint vertex_id: SV_VertexID;
};

struct VSOutput
{
    float4 position: SV_Position;
};

VSOutput vs_main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float x = float(int(input.vertex_id) - 1);
    float y = float(int(input.vertex_id & 1u) * 2 - 1);
    output.position = float4(x, y, 0.0, 1.0);
    return output;
}

struct PSOutput
{
    float4 color: SV_Target0;
};

PSOutput fs_main(VSOutput input)
{
    PSOutput output = (PSOutput)0;
    output.color = float4(1.0, 0.0, 0.0, 1.0);

    return output;
}