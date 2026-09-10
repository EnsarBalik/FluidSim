// Additive optical path through overlapping impostors. OverlapCount writes a
// constant per disc (SebLague/Fluid-Sim); SphereChord writes the Green chord
// length. Camera pass ZTests against copied foam depth so foam punches a hole
// in the thickness field. The sun-space shadow pass keeps ZTest Always.

Shader "Hidden/FluidSim/ParticleThickness"
{
    CGINCLUDE
    #pragma target 4.5
    #include "UnityCG.cginc"
    #include "../Include/ParticleBillboard.hlsl"

    UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

    float _ThicknessContribution;
    int _ThicknessMode;

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

    float frag(Varyings input) : SV_Target
    {
        float radiusSq = dot(input.disc, input.disc);
        if (radiusSq > 1.0)
        {
            discard;
        }

        float z = _ParticleRadius * sqrt(saturate(1.0 - radiusSq));
        float3 viewHit = input.centerVS;
        viewHit.z += z;

        #if !defined(FLUIDSIM_SHADOW_PASS)
        float2 sceneUV = input.screen.xy / input.screen.w;
        float sceneEye = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, sceneUV));
        if (-viewHit.z > sceneEye)
        {
            discard;
        }
        #endif

        if (_ThicknessMode == 1)
        {
            return 2.0 * z;
        }

        return _ThicknessContribution;
    }
    ENDCG

    SubShader
    {
        Tags { "RenderType" = "Transparent" }

        Pass
        {
            Cull Off
            ZWrite Off
            ZTest LEqual
            Blend One One

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_local FLUIDSIM_POSITIONS_2D FLUIDSIM_POSITIONS_3D
            ENDCG
        }

        Pass
        {
            Cull Off
            ZWrite Off
            ZTest Always
            Blend One One

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_local FLUIDSIM_POSITIONS_2D FLUIDSIM_POSITIONS_3D
            #pragma multi_compile_local _ FLUIDSIM_SHADOW_PASS
            ENDCG
        }
    }

    Fallback Off
}
