// SebLague DepthDownsampleCopy: foam RT.g is Unity clip depth.
Shader "Hidden/FluidSim/FoamDepthCopy"
{
    Properties
    {
        _MainTex ("Foam", 2D) = "black" {}
    }

    SubShader
    {
        Cull Off
        ZWrite On
        ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D _MainTex;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings vert(appdata_img input)
            {
                Varyings output;
                output.positionCS = UnityObjectToClipPos(input.vertex);
                output.uv = input.texcoord;
                return output;
            }

            float4 frag(Varyings input, out float outDepth : SV_Depth) : SV_Target
            {
                outDepth = tex2D(_MainTex, input.uv).g;
                return 0.0;
            }
            ENDCG
        }
    }

    Fallback Off
}
