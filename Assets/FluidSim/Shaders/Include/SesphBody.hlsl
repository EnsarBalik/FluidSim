#ifndef FLUIDSIM_SESPH_BODY_INCLUDED
#define FLUIDSIM_SESPH_BODY_INCLUDED

#include "SphGrid.hlsl"
#include "SphKernel.hlsl"

// Weakly compressible SPH, Koschier et al. 2019 Algorithm 1, plus Akinci et al. 2012
// one-layer boundary particles (tutorial §5).
//
//   ComputeBoundaryMasses  m_b = ρ0 / Σ_k W_bk  (missing layers behind the wall become mass)
//   ComputeDensity         fluid kernel sum plus Σ_b m_b W_ib
//   ComputeAccelerations   fluid terms plus pressure mirroring p_b = p_i, ρ_b = ρ0
//   Integrate              symplectic Euler; the box clamp stays as a leak safety net
//
// The boundary grid is static and is built once. Fluid neighbour queries still run over
// the fluid grid; a second one-ring walk covers the wall samples.

#define FS_PARTICLE_GROUP_SIZE 64

RWStructuredBuffer<FS_VEC> _SortedPositions;
RWStructuredBuffer<FS_VEC> _SortedVelocities;
RWStructuredBuffer<FS_VEC> _Positions;
RWStructuredBuffer<FS_VEC> _Velocities;
StructuredBuffer<uint> _SortedIndices;
StructuredBuffer<uint> _CellOffsets;
StructuredBuffer<uint> _CellCounts;
RWStructuredBuffer<float> _Densities;
RWStructuredBuffer<float> _Pressures;
RWStructuredBuffer<float> _Speeds;
RWStructuredBuffer<FS_VEC> _Accelerations;

StructuredBuffer<FS_VEC> _BoundaryPositions;
RWStructuredBuffer<float> _BoundaryMasses;
StructuredBuffer<uint> _BoundaryCellOffsets;
StructuredBuffer<uint> _BoundaryCellCounts;
uint _BoundaryParticleCount;

float _SupportRadius;
float _KernelNormalization;
float _ParticleMass;
float _RestDensity;
float _Stiffness;
float _KinematicViscosity;
float _ParticleRadius;
float _DeltaTime;
float _Restitution;
float4 _Gravity;
float4 _DomainMax;

#if FLUIDSIM_DIMENSIONS == 2
    #define FS_DOMAIN_MAX (_DomainMax.xy)
    #define FS_GRAVITY    (_Gravity.xy)
#else
    #define FS_DOMAIN_MAX (_DomainMax.xyz)
    #define FS_GRAVITY    (_Gravity.xyz)
#endif

void SphCollideAxis(inout float position, inout float velocity, float lower, float upper)
{
    if (position < lower)
    {
        position = lower;
        if (velocity < 0.0)
        {
            velocity = -_Restitution * velocity;
        }
    }
    else if (position > upper)
    {
        position = upper;
        if (velocity > 0.0)
        {
            velocity = -_Restitution * velocity;
        }
    }
}

void SphCollideBox(inout FS_VEC position, inout FS_VEC velocity)
{
    FS_VEC lower = FS_DOMAIN_MIN + _ParticleRadius;
    FS_VEC upper = FS_DOMAIN_MAX - _ParticleRadius;

    SphCollideAxis(position.x, velocity.x, lower.x, upper.x);
    SphCollideAxis(position.y, velocity.y, lower.y, upper.y);
#if FLUIDSIM_DIMENSIONS == 3
    SphCollideAxis(position.z, velocity.z, lower.z, upper.z);
#endif
}

float SphBoundaryKernelSum(FS_VEC position)
{
    if (_BoundaryParticleCount == 0u)
    {
        return 0.0;
    }

    FS_IVEC coord = SphCellCoord(position);
    float sum = 0.0;

    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _BoundaryCellOffsets[cell];
        uint end = start + _BoundaryCellCounts[cell];

        for (uint k = start; k < end; ++k)
        {
            float dist = length(position - _BoundaryPositions[k]);
            sum += SphKernel(dist, _SupportRadius, _KernelNormalization);
        }
    }

    return sum;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ComputeBoundaryMasses(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _BoundaryParticleCount)
    {
        return;
    }

    // Ψ_b = 1 / Σ W, then m_b = ρ0 Ψ_b. On a single layer this sum is too small, so
    // the mass is larger than a fluid particle's — that is the missing-sample correction.
    float kernelSum = SphBoundaryKernelSum(_BoundaryPositions[particle]);
    _BoundaryMasses[particle] = _RestDensity / max(kernelSum, 1e-6);
}

float SphBoundaryDensity(FS_VEC position)
{
    if (_BoundaryParticleCount == 0u)
    {
        return 0.0;
    }

    FS_IVEC coord = SphCellCoord(position);
    float density = 0.0;

    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _BoundaryCellOffsets[cell];
        uint end = start + _BoundaryCellCounts[cell];

        for (uint k = start; k < end; ++k)
        {
            float dist = length(position - _BoundaryPositions[k]);
            density += _BoundaryMasses[k] * SphKernel(dist, _SupportRadius, _KernelNormalization);
        }
    }

    return density;
}

FS_VEC SphBoundaryAcceleration(FS_VEC position, FS_VEC velocity, float pressure, float invDensitySq)
{
    FS_VEC acceleration = FS_ZERO;
    if (_BoundaryParticleCount == 0u)
    {
        return acceleration;
    }

    FS_IVEC coord = SphCellCoord(position);
    float restInvDensitySq = 1.0 / max(_RestDensity * _RestDensity, 1e-12);
    float supportSqEps = 0.01 * _SupportRadius * _SupportRadius;
    float viscScale = 2.0 * (float(FLUIDSIM_DIMENSIONS) + 2.0) * _KinematicViscosity;

    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _BoundaryCellOffsets[cell];
        uint end = start + _BoundaryCellCounts[cell];

        for (uint k = start; k < end; ++k)
        {
            FS_VEC offset = position - _BoundaryPositions[k];
            FS_VEC gradient = SphKernelGradient(offset, _SupportRadius, _KernelNormalization);
            if (dot(gradient, gradient) == 0.0)
            {
                continue;
            }

            float mass = _BoundaryMasses[k];

            // Pressure mirroring: the wall sample is given the fluid particle's own pressure.
            acceleration -= mass * (pressure * invDensitySq + pressure * restInvDensitySq) * gradient;

            float denom = dot(offset, offset) + supportSqEps;
            acceleration += viscScale * (mass / max(_RestDensity, 1e-12)) *
                            (dot(velocity, offset) / denom) * gradient;
        }
    }

    return acceleration;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ComputeDensity(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_IVEC coord = SphCellCoord(position);

    float density = 0.0;
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
            float dist = length(position - _SortedPositions[k]);
            density += _ParticleMass * SphKernel(dist, _SupportRadius, _KernelNormalization);
        }
    }

    density += SphBoundaryDensity(position);

    _Densities[particle] = density;
    _Pressures[particle] = max(_Stiffness * (density / _RestDensity - 1.0), 0.0);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ComputeAccelerations(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_VEC velocity = _SortedVelocities[particle];
    FS_IVEC coord = SphCellCoord(position);

    float density = _Densities[particle];
    float pressure = _Pressures[particle];
    float invDensitySq = 1.0 / max(density * density, 1e-12);
    float supportSqEps = 0.01 * _SupportRadius * _SupportRadius;
    float viscScale = 2.0 * (float(FLUIDSIM_DIMENSIONS) + 2.0) * _KinematicViscosity;

    FS_VEC acceleration = FS_ZERO;
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
            FS_VEC gradient = SphKernelGradient(offset, _SupportRadius, _KernelNormalization);
            if (dot(gradient, gradient) == 0.0)
            {
                continue;
            }

            float neighborDensity = _Densities[k];
            float neighborPressure = _Pressures[k];
            float neighborInvDensitySq = 1.0 / max(neighborDensity * neighborDensity, 1e-12);

            acceleration -= _ParticleMass * (pressure * invDensitySq + neighborPressure * neighborInvDensitySq) *
                            gradient;

            FS_VEC relativeVelocity = velocity - _SortedVelocities[k];
            float denom = dot(offset, offset) + supportSqEps;
            acceleration += viscScale * (_ParticleMass / max(neighborDensity, 1e-12)) *
                            (dot(relativeVelocity, offset) / denom) * gradient;
        }
    }

    acceleration += SphBoundaryAcceleration(position, velocity, pressure, invDensitySq);
    acceleration += FS_GRAVITY;
    _Accelerations[particle] = acceleration;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void Integrate(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC velocity = _SortedVelocities[particle] + _DeltaTime * _Accelerations[particle];
    FS_VEC position = _SortedPositions[particle] + _DeltaTime * velocity;
    SphCollideBox(position, velocity);

    _SortedPositions[particle] = position;
    _SortedVelocities[particle] = velocity;
    _Speeds[particle] = length(velocity);

    uint source = _SortedIndices[particle];
    _Positions[source] = position;
    _Velocities[source] = velocity;
}

#endif // FLUIDSIM_SESPH_BODY_INCLUDED
