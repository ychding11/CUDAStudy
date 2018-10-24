Texture2D destTexture : register( t0 );
SamplerState samLinear: register( s0 );

struct VS_INPUT
{
    float4 Pos : POSITION;
    float2 Tex : TEXCOORD0;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
PS_INPUT VS( VS_INPUT input )
{
	PS_INPUT output;
    output.Pos = input.Pos;
    output.Tex = input.Tex;
    return output;
}

float4 psSampleResultImage( PS_INPUT input) : SV_Target
{
	return destTexture.Sample( samLinear, input.Tex );
}
