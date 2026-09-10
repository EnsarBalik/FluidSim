#ifndef FLUIDSIM_DIFFUSE_BODY_INCLUDED
#define FLUIDSIM_DIFFUSE_BODY_INCLUDED

#include "SphGrid.hlsl"
#include "SphKernel.hlsl"

// Ihmsen, Akinci, Akinci, Teschner, "Unified Spray, Foam and Bubbles
// for Particle-Based Fluids", CGI 2012. Post-process: no forces between
// diffuse particles, so the fluid grid is the only neighbourhood we need.

#define FS_PARTICLE_GROUP_SIZE 64
#define FS_DIFFUSE_KIND_SPRAY 0u
#define FS_DIFFUSE_KIND_FOAM 1u
#define FS_DIFFUSE_KIND_BUBBLE 2u
#define FS_MAX_SPAWN_PER_PARTICLE 2u
#define FS_SPRAY_DRAG 0.04
#define FS_COLLISION_DAMPING 0.1

StructuredBuffer<FS_VEC> _SortedPositions;
StructuredBuffer<FS_VEC> _SortedVelocities;
StructuredBuffer<uint> _CellOffsets;
StructuredBuffer<uint> _CellCounts;

RWStructuredBuffer<float> _SpawnWeight;
RWStructuredBuffer<float3> _DiffusePositions;
RWStructuredBuffer<float3> _DiffuseVelocities;
RWStructuredBuffer<float> _DiffuseLife;
RWStructuredBuffer<uint> _DiffuseKind;
RWStructuredBuffer<uint> _DiffuseOccupied;

float _SupportRadius;
float _KernelNormalization;
float _ParticleRadius;
float _ParticleMass;
float _DeltaTime;
uint _Frame;
uint _MaxDiffuse;
float _PlaneDepth;
float4 _Gravity;
float4 _DomainMinBounds;
float4 _DomainMaxBounds;

float _TrappedAirMin;
float _TrappedAirMax;
float _WaveCrestMin;
float _WaveCrestMax;
float _EnergyMin;
float _EnergyMax;
float _TrappedAirRate;
float _WaveCrestRate;
float _FoamLifeMin;
float _FoamLifeMax;
float _BubbleBuoyancy;
float _BubbleDrag;

float FsPhi(float value, float minThreshold, float maxThreshold)
{
    float range = max(maxThreshold - minThreshold, 1e-5);
    return saturate((min(value, maxThreshold) - min(value, minThreshold)) / range);
}

float FsHatWeight(float dist, float support)
{
    return dist < support ? 1.0 - dist / support : 0.0;
}

uint FsHash(uint x)
{
    x ^= 2747636419u;
    x *= 2654435769u;
    x ^= x >> 16;
    x *= 2654435769u;
    x ^= x >> 16;
    x *= 2654435769u;
    return x;
}

float FsUnit(uint seed)
{
    return (FsHash(seed) & 0x00FFFFFFu) / 16777215.0;
}

float3 FsToWorld(FS_VEC value)
{
#if FLUIDSIM_DIMENSIONS == 2
    return float3(value, _PlaneDepth);
#else
    return value;
#endif
}

FS_VEC FsToFluid(float3 value)
{
#if FLUIDSIM_DIMENSIONS == 2
    return value.xy;
#else
    return value;
#endif
}

float3 FsOrthonormal(float3 axis, float3 hint)
{
    float3 t = abs(axis.y) < 0.9 ? float3(0.0, 1.0, 0.0) : hint;
    return normalize(cross(axis, t));
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void EvaluatePotential(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_VEC velocity = _SortedVelocities[particle];
    FS_IVEC coord = SphCellCoord(position);

    float trapped = 0.0;
    float crest = 0.0;
    FS_VEC normal = FS_ZERO;

    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _CellOffsets[cell];
        uint end = start + _CellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            if (k == particle)
            {
                continue;
            }

            FS_VEC offset = position - _SortedPositions[k];
            float dist = length(offset);
            float hat = FsHatWeight(dist, _SupportRadius);
            if (hat <= 0.0)
            {
                continue;
            }

            FS_VEC relVel = velocity - _SortedVelocities[k];
            float relSpeed = length(relVel);
            if (relSpeed > 1e-5 && dist > 1e-5)
            {
                // Ihmsen eq. (2) is (1 - v̂·x̂): high on impacts (particles
                // approaching). A pushed solid compresses the bow, so that
                // lights up in front of the cube. We want the cavity: high
                // when particles recede, i.e. around and behind the body.
                trapped += relSpeed * (1.0 + dot(relVel / relSpeed, offset / dist)) * hat;
            }

            normal += SphKernelGradient(offset, _SupportRadius, _KernelNormalization);
        }
    }

    float normalLen = length(normal);
    FS_VEC nHat = normalLen > 1e-5 ? normal / normalLen : FS_ZERO;
    float speed = length(velocity);
    FS_VEC vHat = speed > 1e-5 ? velocity / speed : FS_ZERO;

    if (normalLen > 1e-5)
    {
        for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
        {
            FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
            if (!SphCellCoordInBounds(neighborCoord))
            {
                continue;
            }

            uint cell = SphCellIndex(neighborCoord);
            uint start = _CellOffsets[cell];
            uint end = start + _CellCounts[cell];
            for (uint k = start; k < end; ++k)
            {
                if (k == particle)
                {
                    continue;
                }

                FS_VEC offset = position - _SortedPositions[k];
                float dist = length(offset);
                float hat = FsHatWeight(dist, _SupportRadius);
                if (hat <= 0.0 || dist < 1e-5)
                {
                    continue;
                }

                FS_VEC xji = -offset / dist;
                // Ihmsen keeps convex crests (x̂ji·n < 0). That is the bow
                // wave in front of a moving cube. Keep the concave side so
                // foam follows the hollow behind the body.
                if (dot(xji, nHat) < 0.0)
                {
                    continue;
                }

                crest += (1.0 - abs(dot(nHat, offset / dist))) * hat;
            }
        }
    }

    float alongNormal = speed > 1e-5 ? dot(vHat, nHat) : 0.0;
    float crestGate = alongNormal <= -0.6 ? 1.0 : 0.0;
    float kinetic = 0.5 * speed * speed;

    float ita = FsPhi(trapped, _TrappedAirMin, _TrappedAirMax);
    float iwc = FsPhi(crest * crestGate, _WaveCrestMin, _WaveCrestMax);
    float ik = FsPhi(kinetic, _EnergyMin, _EnergyMax);
    _SpawnWeight[particle] = ik * (_TrappedAirRate * ita + _WaveCrestRate * iwc);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void SpawnDiffuse(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    float expected = _SpawnWeight[particle] * _DeltaTime;
    if (expected <= 1e-5)
    {
        return;
    }

    uint seed = FsHash(particle * 747796405u + _Frame * 2891336453u);
    uint count = (uint)floor(expected);
    if (FsUnit(seed) < frac(expected))
    {
        count += 1u;
    }

    count = min(count, FS_MAX_SPAWN_PER_PARTICLE);
    if (count == 0u)
    {
        return;
    }

    float3 position = FsToWorld(_SortedPositions[particle]);
    float3 velocity = FsToWorld(_SortedVelocities[particle]);
#if FLUIDSIM_DIMENSIONS == 2
    velocity.z = 0.0;
#endif

    float speed = length(velocity);
    float3 axis = speed > 1e-4 ? velocity / speed : float3(0.0, 1.0, 0.0);
    float3 e1 = FsOrthonormal(axis, float3(1.0, 0.0, 0.0));
    float3 e2 = normalize(cross(axis, e1));
    // Paper samples xf(t) → xf(t+Δt) along +v, which dumps foam ahead of a
    // pushed cube. Spawn in the particle's wake instead; height is |v|Δt.
    float cylinder = speed * _DeltaTime;
    float3 wakeAxis = -axis;
    float potential = saturate(_SpawnWeight[particle] / max(_TrappedAirRate + _WaveCrestRate, 1.0));
    float life = lerp(_FoamLifeMin, _FoamLifeMax, potential);

    for (uint i = 0u; i < count; ++i)
    {
        uint attemptSeed = seed + i * 1013904223u;
        bool placed = false;
        for (uint attempt = 0u; attempt < 8u; ++attempt)
        {
            uint slot = FsHash(attemptSeed + attempt * 1664525u) % _MaxDiffuse;
            uint previous;
            InterlockedCompareExchange(_DiffuseOccupied[slot], 0u, 1u, previous);
            if (previous != 0u)
            {
                continue;
            }

            float xr = FsUnit(attemptSeed + 17u + attempt);
            float xt = FsUnit(attemptSeed + 31u + attempt);
            float xh = FsUnit(attemptSeed + 53u + attempt);
            float radius = _SupportRadius * sqrt(xr);
            float theta = xt * 6.2831853;
            float3 radial = (cos(theta) * e1 + sin(theta) * e2) * radius;
            float3 spawned = position + radial + wakeAxis * (xh * cylinder);
#if FLUIDSIM_DIMENSIONS == 2
            spawned.z = _PlaneDepth;
#endif
            _DiffusePositions[slot] = spawned;
            _DiffuseVelocities[slot] = velocity + radial;
            _DiffuseLife[slot] = life;
            _DiffuseKind[slot] = FS_DIFFUSE_KIND_FOAM;
            placed = true;
            break;
        }

        if (!placed)
        {
            break;
        }
    }
}

uint FsCountFluidNeighbors(FS_VEC position)
{
    FS_IVEC coord = SphCellCoord(position);
    uint neighbors = 0u;
    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _CellOffsets[cell];
        uint end = start + _CellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            FS_VEC offset = position - _SortedPositions[k];
            if (dot(offset, offset) < _SupportRadius * _SupportRadius)
            {
                neighbors += 1u;
            }
        }
    }

    return neighbors;
}

float3 FsAverageFluidVelocity(FS_VEC position)
{
    FS_IVEC coord = SphCellCoord(position);
    float3 weighted = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;

    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _CellOffsets[cell];
        uint end = start + _CellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            FS_VEC offset = position - _SortedPositions[k];
            float weight = SphKernel(length(offset), _SupportRadius, _KernelNormalization);
            if (weight <= 0.0)
            {
                continue;
            }

            weighted += FsToWorld(_SortedVelocities[k]) * weight;
            weightSum += weight;
        }
    }

    return weightSum > 1e-8 ? weighted / weightSum : float3(0.0, 0.0, 0.0);
}

bool FsOutsideDomain(float3 position)
{
    float3 minBound = _DomainMinBounds.xyz;
    float3 maxBound = _DomainMaxBounds.xyz;
#if FLUIDSIM_DIMENSIONS == 2
    return position.x < minBound.x || position.x > maxBound.x ||
           position.y < minBound.y || position.y > maxBound.y;
#else
    return any(position < minBound) || any(position > maxBound);
#endif
}

void FsKillDiffuse(uint particle)
{
    _DiffuseOccupied[particle] = 0u;
    _DiffuseLife[particle] = 0.0;
}

void FsCollideDomain(inout float3 position, inout float3 velocity)
{
    float3 minBound = _DomainMinBounds.xyz + _ParticleRadius;
    float3 maxBound = _DomainMaxBounds.xyz - _ParticleRadius;
#if FLUIDSIM_DIMENSIONS == 2
    minBound.z = _PlaneDepth;
    maxBound.z = _PlaneDepth;
    velocity.z = 0.0;
#endif

    if (position.x < minBound.x)
    {
        position.x = minBound.x;
        velocity.x = abs(velocity.x) * FS_COLLISION_DAMPING;
    }
    else if (position.x > maxBound.x)
    {
        position.x = maxBound.x;
        velocity.x = -abs(velocity.x) * FS_COLLISION_DAMPING;
    }

    if (position.y < minBound.y)
    {
        position.y = minBound.y;
        velocity.y = abs(velocity.y) * FS_COLLISION_DAMPING;
    }
    else if (position.y > maxBound.y)
    {
        position.y = maxBound.y;
        velocity.y = -abs(velocity.y) * FS_COLLISION_DAMPING;
    }

#if FLUIDSIM_DIMENSIONS == 3
    if (position.z < minBound.z)
    {
        position.z = minBound.z;
        velocity.z = abs(velocity.z) * FS_COLLISION_DAMPING;
    }
    else if (position.z > maxBound.z)
    {
        position.z = maxBound.z;
        velocity.z = -abs(velocity.z) * FS_COLLISION_DAMPING;
    }
#endif
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void AdvectDiffuse(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _MaxDiffuse || _DiffuseOccupied[particle] == 0u)
    {
        return;
    }

    float3 position = _DiffusePositions[particle];
    float3 velocity = _DiffuseVelocities[particle];
    FS_VEC fluidPos = FsToFluid(position);
    uint neighbors = FsCountFluidNeighbors(fluidPos);

#if FLUIDSIM_DIMENSIONS == 2
    uint kind = neighbors < 4u ? FS_DIFFUSE_KIND_SPRAY : (neighbors > 12u ? FS_DIFFUSE_KIND_BUBBLE : FS_DIFFUSE_KIND_FOAM);
#else
    uint kind = neighbors < 6u ? FS_DIFFUSE_KIND_SPRAY : (neighbors > 20u ? FS_DIFFUSE_KIND_BUBBLE : FS_DIFFUSE_KIND_FOAM);
#endif

    float3 gravity = _Gravity.xyz;
    float3 fluidVel = FsAverageFluidVelocity(fluidPos);

    if (kind == FS_DIFFUSE_KIND_SPRAY)
    {
        float sqrSpeed = dot(velocity, velocity);
        float3 drag = sqrSpeed > 1e-8
            ? -normalize(velocity) * sqrSpeed * FS_SPRAY_DRAG
            : float3(0.0, 0.0, 0.0);
        velocity += _DeltaTime * (gravity + drag);
        position += _DeltaTime * velocity;
        if (FsOutsideDomain(position))
        {
            FsKillDiffuse(particle);
            return;
        }
    }
    else if (kind == FS_DIFFUSE_KIND_FOAM)
    {
        velocity = fluidVel;
        position += _DeltaTime * velocity;
        float life = _DiffuseLife[particle] - _DeltaTime;
        if (life <= 0.0)
        {
            FsKillDiffuse(particle);
            return;
        }

        _DiffuseLife[particle] = life;
    }
    else
    {
        float3 accelBuoyancy = gravity * (1.0 - _BubbleBuoyancy);
        float3 accelFluid = (fluidVel - velocity) * _BubbleDrag;
        velocity += _DeltaTime * (accelBuoyancy + accelFluid);
        position += _DeltaTime * velocity;
    }

    FsCollideDomain(position, velocity);
    _DiffusePositions[particle] = position;
    _DiffuseVelocities[particle] = velocity;
    _DiffuseKind[particle] = kind;
}

#endif // FLUIDSIM_DIFFUSE_BODY_INCLUDED
