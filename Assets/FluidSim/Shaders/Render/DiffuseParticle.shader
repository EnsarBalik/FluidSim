// SebLague/Fluid-Sim BillboardFoam.shader, ported to SV_VertexID quads.
// Uses UNITY_MATRIX_VP / V like ParticleDepth so foam and the sheet share clip space.
// rgb = (coverage, unityDepth, linearEyeDepth). Composite lerps to white.

Shader "Hidden/FluidSim/DiffuseParticle"
{
    SubShader
    {
        Tags { "Queue" = "Geometry" "RenderType" = "Opaque" }

        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual
            Blend Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            #include "UnityCG.cginc"

            float2 ParticleCornerOffset(uint corner)
            {
                uint index = corner;
                if (index == 3u) index = 2u;
                else if (index == 4u) index = 1u;
                else if (index == 5u) index = 3u;
                float x = (index & 2u) != 0u ? 1.0 : -1.0;
                float y = (index & 1u) != 0u ? 1.0 : -1.0;
                return float2(x, y);
            }

            float Remap01(float val, float minVal, float maxVal)
            {
                return saturate((val - minVal) / max(maxVal - minVal, 1e-5));
            }

            float LinearDepthToUnityDepth(float linearDepth)
            {
                float depth01 = (linearDepth - _ProjectionParams.y) /
                                max(_ProjectionParams.z - _ProjectionParams.y, 1e-5);
                return (1.0 - (depth01 * _ZBufferParams.y)) / max(depth01 * _ZBufferParams.x, 1e-5);
            }

            StructuredBuffer<float3> _DiffusePositions;
            StructuredBuffer<float3> _DiffuseVelocities;
            StructuredBuffer<float> _DiffuseLife;
            StructuredBuffer<uint> _DiffuseKind;
            StructuredBuffer<uint> _DiffuseOccupied;

            float _DiffuseRadius;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 posWorld : TEXCOORD1;
            };

            v2f vert(uint vertexId : SV_VertexID)
            {
                uint particle = vertexId / 6u;
                v2f o;
                o.pos = float4(2.0, 2.0, 2.0, 1.0);
                o.uv = 0.5;
                o.posWorld = 0;

                if (_DiffuseOccupied[particle] == 0u)
                {
                    return o;
                }

                float2 offset = ParticleCornerOffset(vertexId % 6u);
                float3 worldCentre = _DiffusePositions[particle];

                const float remainingLifetimeDissolveStart = 3.0;
                float dissolveScaleT = saturate(_DiffuseLife[particle] / remainingLifetimeDissolveStart);
                float speed = length(_DiffuseVelocities[particle]);
                float velScale = lerp(0.6, 1.0, Remap01(speed, 1.0, 3.0));
                float particleScale = _DiffuseKind[particle] == 2u ? 0.5 : 1.0;
                float vertScale = _DiffuseRadius * 2.0 * dissolveScaleT * particleScale * velScale;

                float3 camUp = unity_CameraToWorld._m01_m11_m21;
                float3 camRight = unity_CameraToWorld._m00_m10_m20;
                float3 vertPosWorld = worldCentre + camRight * (offset.x * vertScale) +
                                      camUp * (offset.y * vertScale);

                o.pos = mul(UNITY_MATRIX_VP, float4(vertPosWorld, 1.0));
                o.uv = offset * 0.5 + 0.5;
                o.posWorld = worldCentre;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float2 centreOffset = (i.uv - 0.5) * 2.0;
                float sqrDst = dot(centreOffset, centreOffset);
                if (sqrDst > 1.0)
                {
                    discard;
                }

                float linearDepth = abs(mul(UNITY_MATRIX_V, float4(i.posWorld, 1.0)).z);
                return float4(1.0, LinearDepthToUnityDepth(linearDepth), linearDepth, 1.0);
            }
            ENDCG
        }
    }

    Fallback Off
}
