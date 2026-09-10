// Checker floor for the Built-in demo scene, matching SebLague/Fluid-Sim's
// tiled environment. Samples the fluid thickness shadow map so the volume
// darkens the tiles with Beer-Lambert extinction. Unity light shadows still
// apply so the rigid box can sit on the floor.

Shader "Hidden/FluidSim/DemoFloor"
{
    Properties
    {
        _TileCol1 ("Tile A", Color) = (0.82, 0.78, 0.72, 1)
        _TileCol2 ("Tile B", Color) = (0.76, 0.80, 0.84, 1)
        _TileCol3 ("Tile C", Color) = (0.80, 0.74, 0.70, 1)
        _TileCol4 ("Tile D", Color) = (0.72, 0.76, 0.78, 1)
        _TileColVariation ("HSV jitter", Vector) = (0.02, 0.05, 0.06, 0)
        _TileScale ("Tiles per metre", Float) = 1.5
        _TileDarkOffset ("Dark tile value", Float) = -0.18
        _TileOrigin ("Origin", Vector) = (0, 0, 0, 0)
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
            #include "../Include/FluidFloorTiles.hlsl"

            sampler2D _FluidShadowMap;
            float4x4 _FluidShadowVP;
            float3 _FluidShadowExtinction;
            float _FluidShadowAmbient;

            struct Varyings
            {
                float4 pos : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                SHADOW_COORDS(2)
            };

            Varyings vert(appdata_base input)
            {
                Varyings output;
                output.pos = UnityObjectToClipPos(input.vertex);
                output.normalWS = UnityObjectToWorldNormal(input.normal);
                output.worldPos = mul(unity_ObjectToWorld, input.vertex).xyz;
                TRANSFER_SHADOW(output);
                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                float3 tileCol = FluidFloorAlbedo(input.worldPos);

                float4 shadowClip = mul(_FluidShadowVP, float4(input.worldPos, 1.0));
                shadowClip /= max(abs(shadowClip.w), 1e-5);
                float2 shadowUV = shadowClip.xy * 0.5 + 0.5;
                float inMap = shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0
                    ? 1.0 : 0.0;
                float fluidThickness = tex2D(_FluidShadowMap, shadowUV).r * inMap;
                float3 fluidShadow = exp(-fluidThickness * _FluidShadowExtinction);
                float ambient = max(_FluidShadowAmbient, 0.0);
                fluidShadow = fluidShadow * (1.0 - ambient) + ambient;

                float3 normal = normalize(input.normalWS);
                float ndotl = saturate(dot(normal, _WorldSpaceLightPos0.xyz));
                float unityShadow = SHADOW_ATTENUATION(input);
                float3 skyAmbient = ShadeSH9(float4(normal, 1.0));
                float3 lit = tileCol * (_LightColor0.rgb * ndotl * unityShadow * fluidShadow + skyAmbient * 0.35 + 0.08);
                return float4(lit, 1.0);
            }
            ENDCG
        }

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
