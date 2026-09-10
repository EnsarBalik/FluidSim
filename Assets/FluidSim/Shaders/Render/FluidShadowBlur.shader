// Separable Gaussian for the fluid thickness shadow map. Softens the particle
// discs before the floor samples them, matching SebLague's shadowSmooth pass.

Shader "Hidden/FluidSim/FluidShadowBlur"
{
    Properties
    {
        _MainTex ("Shadow", 2D) = "black" {}
    }

    CGINCLUDE
    #include "UnityCG.cginc"

    sampler2D _MainTex;
    float4 _MainTex_TexelSize;
    float2 _BlurDirection;

    struct Varyings
    {
        float4 positionCS : SV_POSITION;
        float2 uv : TEXCOORD0;
    };

    Varyings Vert(appdata_img input)
    {
        Varyings output;
        output.positionCS = UnityObjectToClipPos(input.vertex);
        output.uv = input.texcoord;
        return output;
    }

    float Blur(float2 uv, float2 dir)
    {
        float2 stepUV = dir * _MainTex_TexelSize.xy;
        float sum = tex2D(_MainTex, uv).r * 0.227027;
        sum += tex2D(_MainTex, uv + stepUV).r * 0.1945946;
        sum += tex2D(_MainTex, uv - stepUV).r * 0.1945946;
        sum += tex2D(_MainTex, uv + stepUV * 2.0).r * 0.1216216;
        sum += tex2D(_MainTex, uv - stepUV * 2.0).r * 0.1216216;
        sum += tex2D(_MainTex, uv + stepUV * 3.0).r * 0.070270;
        sum += tex2D(_MainTex, uv - stepUV * 3.0).r * 0.070270;
        return sum;
    }
    ENDCG

    SubShader
    {
        Cull Off
        ZWrite Off
        ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment FragH
            float4 FragH(Varyings input) : SV_Target
            {
                return Blur(input.uv, float2(1.0, 0.0));
            }
            ENDCG
        }

        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment FragV
            float4 FragV(Varyings input) : SV_Target
            {
                return Blur(input.uv, float2(0.0, 1.0));
            }
            ENDCG
        }
    }

    Fallback Off
}
