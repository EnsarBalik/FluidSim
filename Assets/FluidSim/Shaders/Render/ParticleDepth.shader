// Sphere impostors into a camera-distance target. Overlapping particles keep the
// nearest surface via a real depth buffer; empty pixels stay a huge sentinel so
// the filter can tell fluid from air. Simon Green, Screen Space Fluid Rendering,
// GDC 2010; camera-distance encoding matches SebLague/Fluid-Sim.

Shader "Hidden/FluidSim/ParticleDepth"
{
    SubShader
    {
        Tags { "RenderType" = "Opaque" }

        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #pragma multi_compile_local FLUIDSIM_POSITIONS_2D FLUIDSIM_POSITIONS_3D

            #include "UnityCG.cginc"
            #include "../Include/ParticleBillboard.hlsl"

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 centerVS : TEXCOORD0;
                float2 disc : TEXCOORD1;
                float4 screen : TEXCOORD2;
            };

            Varyings vert(uint vertexId : SV_VertexID)
            {
                uint particle = vertexId / 6u;
                float2 offset = ParticleCornerOffset(vertexId % 6u);

                float3 centerVS = mul(UNITY_MATRIX_V, float4(LoadParticlePosition(particle), 1.0)).xyz;
                float3 quadVS = centerVS;
                quadVS.xy += offset * _ParticleRadius;

                Varyings output;
                output.positionCS = mul(UNITY_MATRIX_P, float4(quadVS, 1.0));
                output.centerVS = centerVS;
                output.disc = offset;
                output.screen = ComputeScreenPos(output.positionCS);
                return output;
            }

            float frag(Varyings input, out float outDepth : SV_Depth) : SV_Target
            {
                float radiusSq = dot(input.disc, input.disc);
                if (radiusSq > 1.0)
                {
                    discard;
                }

                // View space looks down -Z, so adding a positive offset moves the
                // sample toward the camera and gives the front of the sphere.
                float z = _ParticleRadius * sqrt(saturate(1.0 - radiusSq));
                float3 viewHit = input.centerVS;
                viewHit.z += z;

                float4 clip = mul(UNITY_MATRIX_P, float4(viewHit, 1.0));
                outDepth = clip.z / clip.w;

                float2 sceneUV = input.screen.xy / input.screen.w;
                float sceneEye = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, sceneUV));
                if (-viewHit.z > sceneEye)
                {
                    discard;
                }

                if (unity_OrthoParams.w > 0.5)
                {
                    return -viewHit.z;
                }

                return length(viewHit);
            }
            ENDCG
        }
    }

    Fallback Off
}
