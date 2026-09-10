// Volume water from SebLague/Fluid-Sim Raymarching.shader + RayMarchingTest.cs.
// Particles are voxelised into a 3D SPH density texture; the isosurface
// (density - densityOffset) is one sheet, not separate discs. Camera rays
// bounce with Fresnel: the more interesting of reflect/refract is traced,
// the other is approximated with the grabbed scene / cube / floor.

Shader "Hidden/FluidSim/FluidComposite"
{
    Properties
    {
        _MainTex ("Packed", 2D) = "black" {}
    }

    SubShader
    {
        Cull Off
        ZWrite Off
        ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            #include "UnityCG.cginc"
            #include "../Include/FluidRayMarch.hlsl"
            #include "../Include/FluidFloorTiles.hlsl"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            sampler2D _FluidSceneColor;
            float4 _FluidSceneColor_TexelSize;
            sampler2D _FluidSceneBlur;
            sampler2D _FluidFoam;
            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

            float4 _AbsorbColor;
            float4 _ScatterColor;
            float3 _LightDirectionVS;
            float3 _DirToSun;
            float _RefractionMultiplier;
            float _SpecularPower;
            float _SpecularIntensity;
            float _FresnelStrength;
            float _ThicknessScale;
            int _DebugView;
            float4x4 _FluidCameraVP;
            float4x4 _FluidInvP;
            float4x4 _FluidCameraToWorld;
            float4x4 _FluidWorldToCamera;
            int _RayMarchSteps;
            float _BackgroundBlur;
            float _BackgroundReveal;
            float _InteriorMix;
            float4x4 _CubeLocalToWorld;
            float4x4 _CubeWorldToLocal;
            float _HasCube;
            float3 _FloorPos;
            float3 _FloorSize;
            float _PlanarFloorEnabled;
            float _SsrEnabled;
            int _SsrSteps;
            float _SsrMaxDistance;
            float _SsrStride;
            float _SsrThickness;
            sampler2D _FluidShadowMap;
            float4x4 _FluidShadowVP;
            float3 _FluidShadowExtinction;
            float _FluidShadowAmbient;

            static const float FluidEmptyThreshold = 1000.0;
            static const float IorAir = 1.0;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 viewVector : TEXCOORD1;
            };

            Varyings vert(appdata_img input)
            {
                Varyings output;
                output.positionCS = UnityObjectToClipPos(input.vertex);
                output.uv = input.texcoord;
                // Seb Raymarching.shader: blit UV → camera ray. _FluidInvP is the
                // GPU texture projection and Y-flips the Game view.
                float3 viewVector = mul(unity_CameraInvProjection, float4(output.uv * 2.0 - 1.0, 0.0, -1.0));
                output.viewVector = mul(unity_CameraToWorld, float4(viewVector, 0.0));
                return output;
            }

            float3 ViewRay(float2 uv)
            {
                return mul(_FluidInvP, float4(uv * 2.0 - 1.0, 0.0, -1.0)).xyz;
            }

            float3 ViewPosition(float2 uv, float cameraDistance)
            {
                float3 viewVector = ViewRay(uv);
                if (unity_OrthoParams.w > 0.5)
                {
                    viewVector.z = -cameraDistance;
                    return viewVector;
                }

                return normalize(viewVector) * cameraDistance;
            }

            float3 WorldViewDir(float2 uv)
            {
                return normalize(mul((float3x3)_FluidCameraToWorld, ViewRay(uv)));
            }

            float3 CameraRayDir(float3 viewVector)
            {
                return normalize(viewVector);
            }

            float SampleDepth(float2 uv, float fallback)
            {
                float depth = tex2D(_MainTex, uv).r;
                return (depth > 1e-5 && depth < FluidEmptyThreshold) ? depth : fallback;
            }

            float3 ReconstructNormal(float2 uv, float centerDepth)
            {
                float2 texel = _MainTex_TexelSize.xy;
                float3 p = ViewPosition(uv, centerDepth);

                float3 ddx = ViewPosition(uv + float2(texel.x, 0.0), SampleDepth(uv + float2(texel.x, 0.0), centerDepth)) - p;
                float3 ddx2 = p - ViewPosition(uv - float2(texel.x, 0.0), SampleDepth(uv - float2(texel.x, 0.0), centerDepth));
                if (abs(ddx.z) > abs(ddx2.z))
                {
                    ddx = ddx2;
                }

                float3 ddy = ViewPosition(uv + float2(0.0, texel.y), SampleDepth(uv + float2(0.0, texel.y), centerDepth)) - p;
                float3 ddy2 = p - ViewPosition(uv - float2(0.0, texel.y), SampleDepth(uv - float2(0.0, texel.y), centerDepth));
                if (abs(ddy.z) > abs(ddy2.z))
                {
                    ddy = ddy2;
                }

                float3 normal = normalize(cross(ddy, ddx));
                if (normal.z > 0.0)
                {
                    normal = -normal;
                }

                return normal;
            }

            float2 WorldToScreenUV(float3 worldPos)
            {
                float4 clip = mul(_FluidCameraVP, float4(worldPos, 1.0));
                if (abs(clip.w) < 1e-5)
                {
                    return float2(-1.0, -1.0);
                }

                return clip.xy / clip.w * 0.5 + 0.5;
            }

            float3 MixFoam(float3 baseCol, float foam)
            {
                return baseCol * (1.0 - foam) + foam;
            }

            float3 SampleSceneAA(float2 uv)
            {
                float2 texel = _FluidSceneColor_TexelSize.xy * 0.7;
                float3 sum = 0.0;
                [unroll]
                for (int ox = -1; ox <= 1; ++ox)
                {
                    [unroll]
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        sum += tex2Dlod(_FluidSceneColor, float4(uv + float2(ox, oy) * texel, 0.0, 0.0)).rgb;
                    }
                }

                return sum / 9.0;
            }

            float3 SampleScene(float2 uv, float blurWeight)
            {
                float3 sharp = tex2D(_FluidSceneColor, uv).rgb;
                float3 blurred = tex2D(_FluidSceneBlur, uv).rgb;
                return lerp(sharp, blurred, saturate(blurWeight));
            }

            float3 SampleSky(float3 dir)
            {
                half4 probe = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, dir, 3.0);
                float3 cubemap = DecodeHDR(probe, unity_SpecCube0_HDR);
                const float3 colGround = float3(0.35, 0.3, 0.35) * 0.53;
                const float3 colSkyHorizon = float3(1.0, 1.0, 1.0);
                const float3 colSkyZenith = float3(0.08, 0.37, 0.73);
                float sun = pow(max(0.0, dot(dir, _DirToSun)), 500.0);
                float skyGradientT = pow(smoothstep(0.0, 0.4, dir.y), 0.35);
                float groundToSkyT = smoothstep(-0.01, 0.0, dir.y);
                float3 skyGradient = lerp(colSkyHorizon, colSkyZenith, skyGradientT);
                float3 procedural = lerp(colGround, skyGradient, groundToSkyT) + sun * (groundToSkyT >= 1.0);
                return cubemap.r + cubemap.g + cubemap.b > 1e-4 ? cubemap : procedural;
            }

            float3 PlanarFloorLit(float3 worldPos)
            {
                float3 albedo = FluidFloorAlbedo(worldPos);
                float4 shadowClip = mul(_FluidShadowVP, float4(worldPos, 1.0));
                shadowClip /= max(abs(shadowClip.w), 1e-5);
                float2 shadowUV = shadowClip.xy * 0.5 + 0.5;
                float inMap = shadowUV.x >= 0.0 && shadowUV.x <= 1.0 &&
                              shadowUV.y >= 0.0 && shadowUV.y <= 1.0
                    ? 1.0 : 0.0;
                float fluidThickness = tex2Dlod(_FluidShadowMap, float4(shadowUV, 0.0, 0.0)).r * inMap;
                float3 fluidShadow = exp(-fluidThickness * _FluidShadowExtinction);
                float ambient = max(_FluidShadowAmbient, 0.0);
                fluidShadow = fluidShadow * (1.0 - ambient) + ambient;
                float ndotl = saturate(dot(float3(0.0, 1.0, 0.0), _DirToSun) * 0.65 + 0.35);
                return albedo * (ndotl * fluidShadow + 0.08);
            }

            bool TraceSsr(float3 origin, float3 dir, out float2 hitUV, out float hitDst)
            {
                hitUV = 0.0;
                hitDst = 1.#INF;
                if (_SsrEnabled < 0.5 || dot(dir, dir) < 0.5)
                {
                    return false;
                }

                int steps = clamp(_SsrSteps, 8, 64);
                float maxDst = max(_SsrMaxDistance, 1.0);
                float stride = max(_SsrStride, 0.03);
                float thickness = max(_SsrThickness, 0.05);
                float2 originUV = WorldToScreenUV(origin);
                float t = stride;

                [loop]
                for (int i = 0; i < 64; ++i)
                {
                    if (i >= steps || t > maxDst)
                    {
                        break;
                    }

                    float3 pos = origin + dir * t;
                    float2 uv = WorldToScreenUV(pos);
                    if (uv.x < 0.002 || uv.x > 0.998 || uv.y < 0.002 || uv.y > 0.998)
                    {
                        return false;
                    }

                    float2 pixelDelta = (uv - originUV) * _ScreenParams.xy;
                    if (dot(pixelDelta, pixelDelta) < 64.0)
                    {
                        t += stride * (1.0 + i * 0.12);
                        continue;
                    }

                    float rayEye = -mul(_FluidWorldToCamera, float4(pos, 1.0)).z;
                    float sceneEye = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv));
                    float delta = rayEye - sceneEye;
                    if (delta > 0.02 && delta < thickness)
                    {
                        hitUV = uv;
                        hitDst = t;
                        return true;
                    }

                    if (delta >= thickness)
                    {
                        hitUV = uv;
                        hitDst = t;
                        return true;
                    }

                    t += stride * (1.0 + i * 0.12);
                }

                return false;
            }

            float SsrEdgeFade(float2 uv)
            {
                float2 e = abs(uv * 2.0 - 1.0);
                return 1.0 - saturate((max(e.x, e.y) - 0.84) / 0.14);
            }

            float3 SampleAnalyticEnvironment(float3 pos, float3 dir, out float fallbackDst)
            {
                HitInfo cubeInfo = (HitInfo)0;
                cubeInfo.dst = 1.#INF;
                if (_HasCube > 0.5)
                {
                    cubeInfo = RayBoxWithMatrix(pos, dir, _CubeLocalToWorld, _CubeWorldToLocal);
                }

                HitInfo floorInfo = (HitInfo)0;
                floorInfo.dst = 1.#INF;
                if (_PlanarFloorEnabled > 0.5)
                {
                    floorInfo = RayBox(pos, dir, _FloorPos, _FloorSize);
                }

                float3 fallback = SampleSky(dir);
                fallbackDst = 1.#INF;
                if (floorInfo.didHit)
                {
                    fallback = PlanarFloorLit(floorInfo.hitPoint);
                    fallbackDst = floorInfo.dst;
                }

                if (cubeInfo.didHit && cubeInfo.dst < fallbackDst)
                {
                    fallbackDst = cubeInfo.dst;
                    float2 cubeUV = WorldToScreenUV(cubeInfo.hitPoint);
                    bool onScreen = cubeUV.x > 0.0 && cubeUV.x < 1.0 && cubeUV.y > 0.0 && cubeUV.y < 1.0;
                    fallback = onScreen
                        ? SampleSceneAA(cubeUV)
                        : saturate(dot(cubeInfo.normal, _DirToSun) * 0.5 + 0.5) * float3(0.95, 0.3, 0.35);
                }

                return fallback;
            }

            float3 SampleEnvironment(float3 pos, float3 dir)
            {
                float fallbackDst;
                float3 fallback = SampleAnalyticEnvironment(pos, dir, fallbackDst);
                float2 ssrUV;
                float ssrDst;
                if (TraceSsr(pos + dir * 0.04, dir, ssrUV, ssrDst) && ssrDst <= fallbackDst + 0.08)
                {
                    return lerp(fallback, SampleSceneAA(ssrUV), SsrEdgeFade(ssrUV));
                }

                return fallback;
            }

            float3 SampleEnvironmentSource(float3 pos, float3 dir)
            {
                float fallbackDst;
                SampleAnalyticEnvironment(pos, dir, fallbackDst);
                float2 ssrUV;
                float ssrDst;
                if (TraceSsr(pos + dir * 0.04, dir, ssrUV, ssrDst) && ssrDst <= fallbackDst + 0.08)
                {
                    return float3(1.0, 0.2, 0.15);
                }

                HitInfo cubeInfo = (HitInfo)0;
                cubeInfo.dst = 1.#INF;
                if (_HasCube > 0.5)
                {
                    cubeInfo = RayBoxWithMatrix(pos, dir, _CubeLocalToWorld, _CubeWorldToLocal);
                }

                HitInfo floorInfo = (HitInfo)0;
                if (_PlanarFloorEnabled > 0.5)
                {
                    floorInfo = RayBox(pos, dir, _FloorPos, _FloorSize);
                }

                if (cubeInfo.didHit && cubeInfo.dst <= fallbackDst + 1e-4)
                {
                    return float3(1.0, 0.85, 0.15);
                }

                if (floorInfo.didHit)
                {
                    return float3(0.2, 1.0, 0.25);
                }

                return float3(0.15, 0.25, 1.0);
            }

            float3 SampleEnvironmentAA(float3 pos, float3 dir)
            {
                float3 right = unity_CameraToWorld._m00_m10_m20;
                float3 up = unity_CameraToWorld._m01_m11_m21;
                float3 sum = 0.0;
                [unroll]
                for (int ox = -1; ox <= 1; ++ox)
                {
                    [unroll]
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        float unusedDst;
                        float3 jitteredFocusPoint = (pos + dir) + (right * ox + up * oy) * 0.7 / _ScreenParams.x;
                        float3 jDir = normalize(jitteredFocusPoint - pos);
                        sum += SampleAnalyticEnvironment(pos, jDir, unusedDst);
                    }
                }

                return sum / 9.0;
            }

            float3 Light(float3 pos, float3 dir)
            {
                float fallbackDst;
                float3 fallback = SampleEnvironmentAA(pos, dir);
                SampleAnalyticEnvironment(pos, dir, fallbackDst);
                float2 ssrUV;
                float ssrDst;
                if (TraceSsr(pos + dir * 0.04, dir, ssrUV, ssrDst) && ssrDst <= fallbackDst + 0.08)
                {
                    return lerp(fallback, SampleSceneAA(ssrUV), SsrEdgeFade(ssrUV));
                }

                return fallback;
            }

            float3 Transmittance(float opticalDepth)
            {
                return FluidTransmittance(opticalDepth, _AbsorbColor.rgb);
            }

            // Seb RayMarchingTest / Raymarching.shader RayMarchFluid.
            float3 RayMarchFluid(float2 uv, float3 rayDir)
            {
                uint rngState = (uint)(uv.x * 1243.0 + uv.y * 96456.0);
                float3 rayPos = _WorldSpaceCameraPos.xyz;
                bool travellingThroughFluid = IsInsideFluid(rayPos);

                float3 transmittance = 1.0;
                float3 light = 0.0;
                int bounces = clamp(_NumRefractions, 1, FluidRefractionLimit);

                [loop]
                for (int i = 0; i < FluidRefractionLimit; ++i)
                {
                    if (i >= bounces)
                    {
                        break;
                    }

                    float densityStepSize = _LightStepSize * (i + 1);
                    bool searchForNextFluidEntryPoint = !travellingThroughFluid;

                    HitInfo cubeHit = (HitInfo)0;
                    cubeHit.dst = 1.#INF;
                    if (_HasCube > 0.5)
                    {
                        cubeHit = RayBoxWithMatrix(rayPos, rayDir, _CubeLocalToWorld, _CubeWorldToLocal);
                    }

                    SurfaceInfo surfaceInfo = FindNextSurface(
                        rayPos, rayDir, searchForNextFluidEntryPoint, rngState, cubeHit.dst);
                    bool useCubeHit = cubeHit.didHit && cubeHit.dst < length(surfaceInfo.pos - rayPos);
                    if (!surfaceInfo.foundSurface)
                    {
                        break;
                    }

                    transmittance *= Transmittance(surfaceInfo.densityAlongRay);

                    if (useCubeHit)
                    {
                        if (travellingThroughFluid)
                        {
                            transmittance *= Transmittance(
                                CalculateDensityAlongRay(cubeHit.hitPoint, cubeHit.normal, densityStepSize));
                        }

                        light += Light(rayPos, rayDir) * transmittance;
                        transmittance = 0.0;
                        break;
                    }

                    if (surfaceInfo.pos.y < _DomainCenter.y - _DomainSize.y * 0.5 + 0.05)
                    {
                        break;
                    }

                    float3 normal = CalculateNormal(surfaceInfo.pos);
                    if (dot(normal, rayDir) > 0.0)
                    {
                        normal = -normal;
                    }

                    float iorA = travellingThroughFluid ? _IndexOfRefraction : IorAir;
                    float iorB = travellingThroughFluid ? IorAir : _IndexOfRefraction;
                    LightResponse lightResponse = CalculateReflectionAndRefraction(rayDir, normal, iorA, iorB);
                    float densityAlongRefractRay = CalculateDensityAlongRay(
                        surfaceInfo.pos, lightResponse.refractDir, densityStepSize);
                    float densityAlongReflectRay = CalculateDensityAlongRay(
                        surfaceInfo.pos, lightResponse.reflectDir, densityStepSize);
                    bool traceRefractedRay = FluidPreferRefractPath(
                        densityAlongRefractRay, lightResponse.refractWeight,
                        densityAlongReflectRay, lightResponse.reflectWeight);
                    travellingThroughFluid = traceRefractedRay != travellingThroughFluid;

                    if (traceRefractedRay)
                    {
                        light += Light(surfaceInfo.pos, lightResponse.reflectDir) * transmittance *
                                 Transmittance(densityAlongReflectRay) * lightResponse.reflectWeight;
                    }
                    else
                    {
                        light += Light(surfaceInfo.pos, lightResponse.refractDir) * transmittance *
                                 Transmittance(densityAlongRefractRay) * lightResponse.refractWeight;
                    }

                    rayPos = surfaceInfo.pos;
                    rayDir = traceRefractedRay ? lightResponse.refractDir : lightResponse.reflectDir;
                    transmittance *= (traceRefractedRay ? lightResponse.refractWeight : lightResponse.reflectWeight);
                }

                float densityRemainder = CalculateDensityAlongRay(rayPos, rayDir, _LightStepSize);
                light += Light(rayPos, rayDir) * transmittance * Transmittance(densityRemainder);
                light += _ScatterColor.rgb * (1.0 - saturate(dot(transmittance, 1.0 / 3.0)));
                return light;
            }

            float4 frag(Varyings input) : SV_Target
            {
                float2 uv = input.uv;
                float4 packedData = tex2D(_MainTex, uv);
                float depthSmooth = packedData.r;
                float thicknessSmooth = packedData.g;
                float thicknessHard = packedData.b;
                float4 scene = tex2D(_FluidSceneColor, uv);
                float4 foamPacked = tex2D(_FluidFoam, uv);
                float foam = foamPacked.r;
                float foamDepth = foamPacked.b;
                float sceneEye = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv));

                bool hasFluid = depthSmooth > 1e-5 && depthSmooth < FluidEmptyThreshold;
                if (_DebugView == 1)
                {
                    return float4(depthSmooth.xxx * 0.08, 1.0);
                }

                if (_DebugView == 2)
                {
                    return float4((thicknessSmooth * _ThicknessScale).xxx * 0.35, 1.0);
                }

                if (_DebugView == 3)
                {
                    if (!hasFluid || -ViewPosition(uv, depthSmooth).z > sceneEye - 0.06)
                    {
                        return float4(0.5, 0.5, 1.0, 1.0);
                    }

                    float3 normalVS = ReconstructNormal(uv, depthSmooth);
                    float3 normalWorld = normalize(mul((float3x3)_FluidCameraToWorld, normalVS));
                    return float4(normalWorld * 0.5 + 0.5, 1.0);
                }

                if (_DebugView == 4)
                {
                    return float4(thicknessHard.xxx * 0.35, 1.0);
                }

                if (_DebugView == 5)
                {
                    return float4(foam.xxx, 1.0);
                }

                float3 rayDir = CameraRayDir(input.viewVector);
                uint rngState = (uint)(uv.x * 1243.0 + uv.y * 96456.0);
                SurfaceInfo firstHit = FindNextSurface(
                    _WorldSpaceCameraPos.xyz, rayDir, !IsInsideFluid(_WorldSpaceCameraPos.xyz),
                    rngState, 1.#INF);

                if (_DebugView == 6)
                {
                    if (!firstHit.foundSurface)
                    {
                        return scene;
                    }

                    float3 normal = CalculateNormal(firstHit.pos);
                    if (dot(normal, rayDir) > 0.0)
                    {
                        normal = -normal;
                    }

                    LightResponse bounce = CalculateReflectionAndRefraction(
                        rayDir, normal, IorAir, _IndexOfRefraction);
                    return float4(SampleEnvironmentSource(firstHit.pos, bounce.reflectDir), 1.0);
                }

                float3 water;
                if (!firstHit.foundSurface)
                {
                    // Volume miss: keep the grabbed camera image. Replacing every
                    // pixel with a reconstructed environment flipped the Game view
                    // and hid the tank / particles, leaving only foam.
                    if (hasFluid)
                    {
                        float sheet = max(thicknessSmooth * _ThicknessScale, 0.0);
                        float3 transmission = FluidTransmittance(sheet, _AbsorbColor.rgb);
                        transmission = max(transmission, _BackgroundReveal);
                        water = scene.rgb * transmission + _ScatterColor.rgb * (1.0 - transmission);
                    }
                    else
                    {
                        water = scene.rgb;
                    }
                }
                else
                {
                    water = RayMarchFluid(uv, rayDir);
                }

                float fluidDst = hasFluid ? depthSmooth : 1e8;
                if (foam > 1e-4 && foamDepth <= fluidDst)
                {
                    water = MixFoam(water, foam);
                }

                return float4(water, 1.0);
            }
            ENDCG
        }
    }

    Fallback Off
}
