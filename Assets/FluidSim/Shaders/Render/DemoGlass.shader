Shader "Hidden/FluidSim/DemoGlass"
{
    Properties
    {
        _Color ("Color", Color) = (0.65, 0.82, 0.92, 0.12)
    }

    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
        Cull Off
        ZWrite Off
        ZTest LEqual
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            float4 _Color;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                float3 viewWS : TEXCOORD1;
            };

            Varyings vert(float4 position : POSITION, float3 normal : NORMAL)
            {
                Varyings output;
                float3 world = mul(unity_ObjectToWorld, position).xyz;
                output.positionCS = UnityObjectToClipPos(position);
                output.normalWS = UnityObjectToWorldNormal(normal);
                output.viewWS = _WorldSpaceCameraPos - world;
                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                float3 normal = normalize(input.normalWS);
                float3 view = normalize(input.viewWS);
                float fresnel = pow(1.0 - saturate(abs(dot(normal, view))), 3.0);
                float alpha = saturate(_Color.a + fresnel * 0.35);
                float3 color = lerp(_Color.rgb, float3(1.0, 1.0, 1.0), fresnel);
                return float4(color, alpha);
            }
            ENDCG
        }
    }

    Fallback Off
}
