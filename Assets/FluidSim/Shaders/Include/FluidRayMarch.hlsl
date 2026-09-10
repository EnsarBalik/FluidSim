#ifndef FLUIDSIM_RAYMARCH_INCLUDED
#define FLUIDSIM_RAYMARCH_INCLUDED

// Volume walk from SebLague/Fluid-Sim Raymarching.shader.
// SampleDensity reads the SPH Texture3D (not a screen-space particle sheet),
// so clustered particles become one isosurface. FindNextSurface + Fresnel
// bounces replace per-blob screen-space reflection.

static const float FluidRayTinyNudge = 0.01;
static const int FluidViewMarchLimit = 512;
static const int FluidLightMarchLimit = 64;
static const int FluidRefractionLimit = 8;

Texture3D<float> DensityMap;
SamplerState sampler_DensityMap;

float3 _DomainCenter;
float3 _DomainSize;
float _DensityMultiplier;
float _RayMarchStep;
float _DensityOffset;
float _LightStepSize;
float _IndexOfRefraction;
int _NumRefractions;

struct HitInfo
{
    bool didHit;
    bool isInside;
    float dst;
    float3 hitPoint;
    float3 normal;
};

struct SurfaceInfo
{
    float3 pos;
    float3 normal;
    float densityAlongRay;
    bool foundSurface;
};

struct LightResponse
{
    float3 reflectDir;
    float3 refractDir;
    float reflectWeight;
    float refractWeight;
};

float2 FluidRayBoxDst(float3 boundsMin, float3 boundsMax, float3 rayOrigin, float3 rayDir)
{
    float3 invRayDir = 1.0 / rayDir;
    float3 t0 = (boundsMin - rayOrigin) * invRayDir;
    float3 t1 = (boundsMax - rayOrigin) * invRayDir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);

    float dstA = max(max(tmin.x, tmin.y), tmin.z);
    float dstB = min(min(tmax.x, tmax.y), tmax.z);
    float dstToBox = max(0.0, dstA);
    float dstInsideBox = max(0.0, dstB - dstToBox);
    return float2(dstToBox, dstInsideBox);
}

float2 RayBoxDst(float3 boundsMin, float3 boundsMax, float3 rayOrigin, float3 rayDir)
{
    return FluidRayBoxDst(boundsMin, boundsMax, rayOrigin, rayDir);
}

HitInfo RayUnitBox(float3 pos, float3 dir)
{
    const float3 boxMin = float3(-1.0, -1.0, -1.0);
    const float3 boxMax = float3(1.0, 1.0, 1.0);
    float3 invDir = 1.0 / dir;
    float3 tMin = (boxMin - pos) * invDir;
    float3 tMax = (boxMax - pos) * invDir;
    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);

    HitInfo hitInfo = (HitInfo)0;
    hitInfo.dst = 1.#INF;
    hitInfo.didHit = tFar >= tNear && tFar > 0.0;
    hitInfo.isInside = tFar > tNear && tNear <= 0.0;
    if (!hitInfo.didHit)
    {
        return hitInfo;
    }

    float hitDst = hitInfo.isInside ? tFar : tNear;
    float3 hitPos = pos + dir * hitDst;
    hitInfo.dst = hitDst;
    hitInfo.hitPoint = hitPos;

    float3 o = 1.0 - abs(hitPos);
    float3 absNormal = (o.x < o.y && o.x < o.z)
        ? float3(1.0, 0.0, 0.0)
        : (o.y < o.z) ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    hitInfo.normal = absNormal * sign(hitPos) * (hitInfo.isInside ? -1.0 : 1.0);
    return hitInfo;
}

HitInfo FluidRayUnitBox(float3 pos, float3 dir)
{
    return RayUnitBox(pos, dir);
}

HitInfo RayBox(float3 rayPos, float3 rayDir, float3 centre, float3 size)
{
    HitInfo hitInfo = RayUnitBox((rayPos - centre) / size, rayDir / size);
    hitInfo.hitPoint = hitInfo.hitPoint * size + centre;
    if (hitInfo.didHit)
    {
        hitInfo.dst = length(hitInfo.hitPoint - rayPos);
    }

    return hitInfo;
}

HitInfo RayBoxWithMatrix(float3 rayPos, float3 rayDir, float4x4 localToWorld, float4x4 worldToLocal)
{
    float3 posLocal = mul(worldToLocal, float4(rayPos, 1.0)).xyz;
    float3 dirLocal = mul(worldToLocal, float4(rayDir, 0.0)).xyz;
    HitInfo hitInfo = RayUnitBox(posLocal, dirLocal);
    hitInfo.normal = normalize(mul(localToWorld, float4(hitInfo.normal, 0.0)).xyz);
    hitInfo.hitPoint = mul(localToWorld, float4(hitInfo.hitPoint, 1.0)).xyz;
    if (hitInfo.didHit)
    {
        hitInfo.dst = length(hitInfo.hitPoint - rayPos);
    }

    return hitInfo;
}

float3 FluidTransmittance(float opticalDepth, float3 extinction)
{
    return exp(-opticalDepth * extinction);
}

float SampleDensity(float3 pos)
{
    float3 uvw = (pos - _DomainCenter) / _DomainSize + 0.5;
    const float epsilon = 0.0001;
    bool isEdge = any(uvw >= 1.0 - epsilon) || any(uvw <= epsilon);
    if (isEdge)
    {
        return -_DensityOffset;
    }

    return DensityMap.SampleLevel(sampler_DensityMap, uvw, 0) - _DensityOffset;
}

float CalculateDensityAlongRay(float3 rayPos, float3 rayDir, float stepSize)
{
    if (dot(rayDir, rayDir) < 0.9)
    {
        return 0.0;
    }

    float3 domainMin = _DomainCenter - _DomainSize * 0.5;
    float3 domainMax = _DomainCenter + _DomainSize * 0.5;
    float2 boundsDstInfo = RayBoxDst(domainMin, domainMax, rayPos, rayDir);
    float dstThroughBounds = boundsDstInfo.y;
    if (dstThroughBounds <= 0.0)
    {
        return 0.0;
    }

    float dstTravelled = 0.0;
    float opticalDepth = 0.0;
    float nudge = stepSize * 0.5;
    float3 entryPoint = rayPos + rayDir * (boundsDstInfo.x + nudge);
    dstThroughBounds -= nudge + FluidRayTinyNudge;

    [loop]
    for (int i = 0; i < FluidLightMarchLimit; ++i)
    {
        if (dstTravelled >= dstThroughBounds)
        {
            break;
        }

        float3 samplePos = entryPoint + rayDir * dstTravelled;
        float density = SampleDensity(samplePos) * _DensityMultiplier * stepSize;
        if (density > 0.0)
        {
            opticalDepth += density;
        }

        dstTravelled += stepSize;
    }

    return opticalDepth;
}

float3 CalculateClosestFaceNormal(float3 boxSize, float3 p)
{
    float3 halfSize = boxSize * 0.5;
    float3 o = halfSize - abs(p);
    return (o.x < o.y && o.x < o.z)
        ? float3(sign(p.x), 0.0, 0.0)
        : (o.y < o.z) ? float3(0.0, sign(p.y), 0.0) : float3(0.0, 0.0, sign(p.z));
}

float3 CalculateNormal(float3 pos)
{
    const float s = 0.1;
    float dx = SampleDensity(pos - float3(s, 0.0, 0.0)) - SampleDensity(pos + float3(s, 0.0, 0.0));
    float dy = SampleDensity(pos - float3(0.0, s, 0.0)) - SampleDensity(pos + float3(0.0, s, 0.0));
    float dz = SampleDensity(pos - float3(0.0, 0.0, s)) - SampleDensity(pos + float3(0.0, 0.0, s));
    float3 volumeNormal = normalize(float3(dx, dy, dz) + 1e-6);

    float3 o = _DomainSize * 0.5 - abs(pos - _DomainCenter);
    float faceWeight = min(o.x, min(o.y, o.z));
    float3 faceNormal = CalculateClosestFaceNormal(_DomainSize, pos - _DomainCenter);
    const float smoothDst = 0.3;
    const float smoothPow = 5.0;
    faceWeight = (1.0 - smoothstep(0.0, smoothDst, faceWeight)) *
                 (1.0 - pow(saturate(volumeNormal.y), smoothPow));
    return normalize(volumeNormal * (1.0 - faceWeight) + faceNormal * faceWeight);
}

bool IsInsideFluid(float3 pos)
{
    float3 domainMin = _DomainCenter - _DomainSize * 0.5;
    float3 domainMax = _DomainCenter + _DomainSize * 0.5;
    float2 boundsDstInfo = RayBoxDst(domainMin, domainMax, pos, float3(0.0, 0.0, 1.0));
    return boundsDstInfo.x <= 0.0 && boundsDstInfo.y > 0.0 && SampleDensity(pos) > 0.0;
}

uint FluidNextRandom(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint result = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result;
}

float FluidRandomValue(inout uint state)
{
    return FluidNextRandom(state) / 4294967295.0;
}

SurfaceInfo FindNextSurface(
    float3 origin, float3 rayDir, bool findNextFluidEntryPoint, uint rngState, float maxDst)
{
    SurfaceInfo info = (SurfaceInfo)0;
    if (dot(rayDir, rayDir) < 0.5)
    {
        return info;
    }

    float3 domainMin = _DomainCenter - _DomainSize * 0.5;
    float3 domainMax = _DomainCenter + _DomainSize * 0.5;
    float2 boundsDstInfo = RayBoxDst(domainMin, domainMax, origin, rayDir);
    float r = (FluidRandomValue(rngState) - 0.5) * _RayMarchStep * 0.4;
    bool hasExittedFluid = !IsInsideFluid(origin);
    origin = origin + rayDir * (boundsDstInfo.x + r);

    float stepSize = max(_RayMarchStep, 0.02);
    bool hasEnteredFluid = false;
    float3 lastPosInFluid = origin;
    float dstToTest = boundsDstInfo.y - FluidRayTinyNudge * 2.0;

    [loop]
    for (int i = 0; i < FluidViewMarchLimit; ++i)
    {
        float dst = i * stepSize;
        if (dst >= dstToTest)
        {
            break;
        }

        bool isLastStep = dst + stepSize >= dstToTest || i + 1 >= FluidViewMarchLimit;
        float3 samplePos = origin + rayDir * dst;
        float density = SampleDensity(samplePos);
        bool insideFluid = density > 0.0;
        float thickness = density * _DensityMultiplier * stepSize;
        if (insideFluid)
        {
            hasEnteredFluid = true;
            lastPosInFluid = samplePos;
            if (dst <= maxDst)
            {
                info.densityAlongRay += thickness;
            }
        }

        if (!insideFluid)
        {
            hasExittedFluid = true;
        }

        bool found = findNextFluidEntryPoint
            ? insideFluid && hasExittedFluid
            : hasEnteredFluid && (!insideFluid || isLastStep);

        if (found)
        {
            info.pos = lastPosInFluid;
            info.foundSurface = true;
            break;
        }
    }

    return info;
}

float CalculateReflectance(float3 inDir, float3 normal, float iorA, float iorB)
{
    float refractRatio = iorA / iorB;
    float cosAngleIn = -dot(inDir, normal);
    float sinSqrAngleOfRefraction = refractRatio * refractRatio * (1.0 - cosAngleIn * cosAngleIn);
    if (sinSqrAngleOfRefraction >= 1.0)
    {
        return 1.0;
    }

    float cosAngleOfRefraction = sqrt(1.0 - sinSqrAngleOfRefraction);
    float rPerp = (iorA * cosAngleIn - iorB * cosAngleOfRefraction) /
                  (iorA * cosAngleIn + iorB * cosAngleOfRefraction);
    float rPar = (iorB * cosAngleIn - iorA * cosAngleOfRefraction) /
                 (iorB * cosAngleIn + iorA * cosAngleOfRefraction);
    return (rPerp * rPerp + rPar * rPar) * 0.5;
}

float3 RefractDir(float3 inDir, float3 normal, float iorA, float iorB)
{
    float refractRatio = iorA / iorB;
    float cosAngleIn = -dot(inDir, normal);
    float sinSqrAngleOfRefraction = refractRatio * refractRatio * (1.0 - cosAngleIn * cosAngleIn);
    if (sinSqrAngleOfRefraction > 1.0)
    {
        return 0.0;
    }

    return refractRatio * inDir + (refractRatio * cosAngleIn - sqrt(1.0 - sinSqrAngleOfRefraction)) * normal;
}

LightResponse CalculateReflectionAndRefraction(float3 inDir, float3 normal, float iorA, float iorB)
{
    LightResponse result;
    result.reflectWeight = CalculateReflectance(inDir, normal, iorA, iorB);
    result.refractWeight = 1.0 - result.reflectWeight;
    result.reflectDir = inDir - 2.0 * dot(inDir, normal) * normal;
    result.refractDir = RefractDir(inDir, normal, iorA, iorB);
    return result;
}

bool FluidPreferRefractPath(float densityRefract, float refractWeight, float densityReflect, float reflectWeight)
{
    return densityRefract * refractWeight > densityReflect * reflectWeight;
}

#endif
