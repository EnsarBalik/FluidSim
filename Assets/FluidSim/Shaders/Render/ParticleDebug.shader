// Draws particles straight from the GPU buffers as camera-facing discs, coloured by an
// arbitrary per-particle scalar. This is diagnostic output, not the fluid renderer: colouring
// by neighbour count is the fastest way to see whether the neighbour grid is intact, and the
// same path takes density or pressure once the solver exists.

Shader "Hidden/FluidSim/ParticleDebug"
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

            // The position buffer's stride differs between 2D and 3D, so the declaration has
            // to follow the simulation. The renderer enables the matching keyword.
            #pragma multi_compile_local FLUIDSIM_POSITIONS_2D FLUIDSIM_POSITIONS_3D

            #include "UnityCG.cginc"

            StructuredBuffer<float> _Values;

            float _ParticleRadius;
            float _PlaneDepth;
            float4 _ValueRange; // x = minimum, y = maximum

            #if defined(FLUIDSIM_POSITIONS_3D)
                StructuredBuffer<float3> _Positions;

                float3 LoadPosition(uint particle)
                {
                    return _Positions[particle];
                }
            #else
                StructuredBuffer<float2> _Positions;

                float3 LoadPosition(uint particle)
                {
                    return float3(_Positions[particle], _PlaneDepth);
                }
            #endif

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 disc : TEXCOORD0;
                float3 color : TEXCOORD1;
            };

            // Two triangles over the corners of a unit quad, addressed without a lookup table
            // so there is no dynamically indexed constant array in the vertex shader.
            float2 CornerOffset(uint corner)
            {
                uint index = corner;
                if (index == 3u)
                {
                    index = 2u;
                }
                else if (index == 4u)
                {
                    index = 1u;
                }
                else if (index == 5u)
                {
                    index = 3u;
                }

                float x = (index & 2u) != 0u ? 1.0 : -1.0;
                float y = (index & 1u) != 0u ? 1.0 : -1.0;
                return float2(x, y);
            }

            float3 Ramp(float t)
            {
                t = saturate(t);

                const float3 cold = float3(0.09, 0.16, 0.45);
                const float3 mid = float3(0.13, 0.72, 0.56);
                const float3 warm = float3(0.97, 0.80, 0.22);
                const float3 hot = float3(0.84, 0.16, 0.16);

                float3 color = lerp(cold, mid, saturate(t * 3.0));
                color = lerp(color, warm, saturate(t * 3.0 - 1.0));
                color = lerp(color, hot, saturate(t * 3.0 - 2.0));
                return color;
            }

            Varyings vert(uint vertexId : SV_VertexID)
            {
                uint particle = vertexId / 6u;
                float2 offset = CornerOffset(vertexId % 6u);

                // Expanding the quad in view space rather than world space keeps the disc facing
                // the camera, which is what makes the same shader usable for an orthographic 2D
                // view and a freely orbiting 3D one.
                float3 view = mul(UNITY_MATRIX_V, float4(LoadPosition(particle), 1.0)).xyz;
                view.xy += offset * _ParticleRadius;

                float span = max(_ValueRange.y - _ValueRange.x, 1e-5);
                float normalized = (_Values[particle] - _ValueRange.x) / span;

                Varyings output;
                output.positionCS = mul(UNITY_MATRIX_P, float4(view, 1.0));
                output.disc = offset;
                output.color = Ramp(normalized);
                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                float radiusSq = dot(input.disc, input.disc);
                if (radiusSq > 1.0)
                {
                    discard;
                }

                // Fake sphere shading so overlapping particles stay individually readable.
                float shade = sqrt(saturate(1.0 - radiusSq));
                return float4(input.color * (0.45 + 0.55 * shade), 1.0);
            }
            ENDCG
        }
    }

    Fallback Off
}
