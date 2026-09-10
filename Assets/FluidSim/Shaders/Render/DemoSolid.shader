Shader "Hidden/FluidSim/DemoSolid"
{
    Properties
    {
        _Color ("Color", Color) = (0.92, 0.45, 0.16, 1)
    }

    SubShader
    {
        Tags { "Queue" = "Geometry" "RenderType" = "Opaque" }
        Cull Back

        Pass
        {
            Tags { "LightMode" = "ForwardBase" }
            ZWrite On
            ZTest LEqual

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fwdbase
            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"

            float4 _Color;

            struct Varyings
            {
                float4 pos : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                SHADOW_COORDS(1)
            };

            Varyings vert(appdata_base input)
            {
                Varyings output;
                output.pos = UnityObjectToClipPos(input.vertex);
                output.normalWS = UnityObjectToWorldNormal(input.normal);
                TRANSFER_SHADOW(output);
                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                float3 normal = normalize(input.normalWS);
                float ndotl = saturate(dot(normal, _WorldSpaceLightPos0.xyz));
                float shadow = SHADOW_ATTENUATION(input);
                float3 ambient = ShadeSH9(float4(normal, 1.0));
                float3 lit = _Color.rgb * (_LightColor0.rgb * ndotl * shadow + ambient * 0.35 + 0.12);
                return float4(lit, 1.0);
            }
            ENDCG
        }

        // Built-in _CameraDepthTexture is filled from this pass. Without it the
        // fluid composite thinks the box is missing and paints water over it.
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }
            ZWrite On
            ZTest LEqual
            ColorMask 0

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_shadowcaster
            #include "UnityCG.cginc"

            struct Varyings
            {
                V2F_SHADOW_CASTER;
            };

            Varyings vert(appdata_base input)
            {
                Varyings output;
                TRANSFER_SHADOW_CASTER_NORMALOFFSET(output)
                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                SHADOW_CASTER_FRAGMENT(input)
            }
            ENDCG
        }
    }

    Fallback Off
}
