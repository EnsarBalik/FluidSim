// Pack raw depth + thickness, then a separable world-space bilateral.
// Kernel width is a world radius converted to pixels at the sample's camera
// distance (Freya Holmér), so the sheet stays a constant thickness in metres.
// Range Gaussian uses the unsmoothed depth in A so silhouettes do not bleed.

Shader "Hidden/FluidSim/FluidDepthFilter"
{
    Properties
    {
        _MainTex ("Packed", 2D) = "black" {}
    }

    CGINCLUDE
    #include "UnityCG.cginc"

    sampler2D _MainTex;
    float4 _MainTex_TexelSize;
    sampler2D _FluidRawDepth;
    sampler2D _FluidRawThickness;

    float _WorldFilterRadius;
    int _MaxScreenSpaceRadius;
    float _FilterStrength;
    float _FilterDiffStrength;
    float _FilterBilateral;
    float _CameraProjectionM00;

    static const float FluidEmptyThreshold = 1000.0;
    static const int FilterLoopLimit = 32;

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

    float ScreenSpaceRadius(float worldRadius, float depth, float imageWidth)
    {
        float widthScale = _CameraProjectionM00;
        float pxPerMeter = (imageWidth * widthScale) / (2.0 * max(depth, 1e-4));
        return abs(pxPerMeter) * worldRadius;
    }

    float4 Blur1D(float2 uv, float2 dir)
    {
        float4 original = tex2D(_MainTex, uv);
        float depth = original.a;
        if (depth <= 1e-5 || depth > FluidEmptyThreshold)
        {
            return original;
        }

        float radiusFloat = ScreenSpaceRadius(_WorldFilterRadius, depth, _MainTex_TexelSize.z);
        int radius = (int)ceil(radiusFloat);
        if (radius <= 1 && _WorldFilterRadius > 0.0)
        {
            radius = 2;
        }

        radius = min(radius, min(_MaxScreenSpaceRadius, FilterLoopLimit));
        float fractional = max(0.0, radius - radiusFloat);
        float sigma = max(1e-7, (radius - fractional) / (6.0 * max(_FilterStrength, 1e-3)));
        float2 texelDelta = _MainTex_TexelSize.xy * dir;

        float4 sum = 0.0;
        float weightSum = 0.0;
        for (int x = -32; x <= 32; ++x)
        {
            if (x < -radius || x > radius)
            {
                continue;
            }

            float2 uv2 = uv + texelDelta * x;
            float4 sampleValue = tex2Dlod(_MainTex, float4(uv2, 0.0, 0.0));
            if (sampleValue.a <= 1e-5 || sampleValue.a > FluidEmptyThreshold)
            {
                continue;
            }

            float spatial = exp(-x * x / (2.0 * sigma * sigma));
            float centreDiff = original.a - sampleValue.a;
            // FluidRenderTest BlurType: Bilateral keeps silhouettes; Gaussian
            // (range weight = 1) is the smoother sheet Seb exposes as an option.
            float rangeWeight = _FilterBilateral > 0.5
                ? exp(-centreDiff * centreDiff * _FilterDiffStrength)
                : 1.0;
            float weight = spatial * rangeWeight;
            sum += sampleValue * weight;
            weightSum += weight;
        }

        if (weightSum <= 1e-5)
        {
            return original;
        }

        sum /= weightSum;
        // R/G = smoothed depth/thickness; B stays raw thickness; A stays raw depth.
        return float4(lerp(original.rgb, sum.rgb, float3(1.0, 1.0, 0.0)), original.a);
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
            #pragma fragment FragPack
            #pragma target 4.5

            float4 FragPack(Varyings input) : SV_Target
            {
                float depth = tex2D(_FluidRawDepth, input.uv).r;
                float thick = tex2D(_FluidRawThickness, input.uv).r;
                return float4(depth, thick, thick, depth);
            }
            ENDCG
        }

        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment FragH
            #pragma target 4.5

            float4 FragH(Varyings input) : SV_Target
            {
                return Blur1D(input.uv, float2(1.0, 0.0));
            }
            ENDCG
        }

        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment FragV
            #pragma target 4.5

            float4 FragV(Varyings input) : SV_Target
            {
                return Blur1D(input.uv, float2(0.0, 1.0));
            }
            ENDCG
        }
    }

    Fallback Off
}
