#ifndef FLUIDSIM_DFSPH_BODY_INCLUDED
#define FLUIDSIM_DFSPH_BODY_INCLUDED

#include "SphGrid.hlsl"
#include "SphKernel.hlsl"

// Divergence-free SPH, Koschier et al. 2019 §4.10 / Algorithms 4–6.
//
// Split across a neighbourhood rebuild:
//   BeginStep   density, k, non-pressure predict, constant-density iterations, x += dt v*
//   (host rebuilds the fluid grid at the new positions)
//   EndStep     density, k, divergence-free iterations, v = v*
//
// Boundary handling is the same Akinci one-layer scheme as the SESPH solver.
// Dynamic rigid samples are a second Akinci stream (Akinci et al. 2012) and
// receive the equal-opposite force of every fluid pair they participate in.
// Implicit viscosity (Weiler 2018, Jacobi) replaces the explicit Brookshaw
// term when _ImplicitViscosityIterations > 0.

#define FS_PARTICLE_GROUP_SIZE 64

RWStructuredBuffer<FS_VEC> _SortedPositions;
RWStructuredBuffer<FS_VEC> _SortedVelocities;
RWStructuredBuffer<FS_VEC> _Positions;
RWStructuredBuffer<FS_VEC> _Velocities;
RWStructuredBuffer<FS_VEC> _PredictedVelocities;
RWStructuredBuffer<FS_VEC> _ViscosityRhs;
RWStructuredBuffer<FS_VEC> _ViscosityVelocities;
StructuredBuffer<uint> _SortedIndices;
StructuredBuffer<uint> _CellOffsets;
StructuredBuffer<uint> _CellCounts;
RWStructuredBuffer<float> _Densities;
RWStructuredBuffer<float> _PredictedDensities;
RWStructuredBuffer<float> _Pressures;
RWStructuredBuffer<float> _StiffnessFactors;
RWStructuredBuffer<float> _Speeds;
RWStructuredBuffer<FS_VEC> _SurfaceNormals;
RWStructuredBuffer<float3> _AngularVelocities;
RWStructuredBuffer<float3> _SortedAngularVelocities;

StructuredBuffer<FS_VEC> _BoundaryPositions;
RWStructuredBuffer<float> _BoundaryMasses;
StructuredBuffer<uint> _BoundaryCellOffsets;
StructuredBuffer<uint> _BoundaryCellCounts;
uint _BoundaryParticleCount;

StructuredBuffer<FS_VEC> _RigidPositions;
StructuredBuffer<FS_VEC> _RigidVelocities;
StructuredBuffer<float> _RigidMasses;
RWStructuredBuffer<float> _RigidMassesOut;
StructuredBuffer<uint> _RigidCellOffsets;
StructuredBuffer<uint> _RigidCellCounts;
RWStructuredBuffer<float3> _RigidForces;
RWStructuredBuffer<float> _UnsortedRigidMasses;
StructuredBuffer<uint> _RigidSortedIndices;
RWStructuredBuffer<float3> _RigidReduced;
float4 _RigidCenterOfMass;
uint _RigidParticleCount;
int _RigidAccumulateMode;

float _SupportRadius;
float _KernelNormalization;
float _ParticleMass;
float _RestDensity;
float _KinematicViscosity;
float _ImplicitViscosityIterations;
float _SurfaceTension;
float _Adhesion;
float _Vorticity;
float _ViscosityOmega;
float _InertiaInverse;
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
            sum += SphKernel(length(position - _BoundaryPositions[k]), _SupportRadius, _KernelNormalization);
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
            density += _BoundaryMasses[k] *
                       SphKernel(length(position - _BoundaryPositions[k]), _SupportRadius, _KernelNormalization);
        }
    }

    return density;
}

float SphAkinciAdhesion(float distance, float support);

float3 SphToFloat3(FS_VEC value)
{
#if FLUIDSIM_DIMENSIONS == 2
    return float3(value, 0.0);
#else
    return value;
#endif
}

float SphRigidDensity(FS_VEC position)
{
    if (_RigidParticleCount == 0u)
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
        uint start = _RigidCellOffsets[cell];
        uint end = start + _RigidCellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            density += _RigidMasses[k] *
                       SphKernel(length(position - _RigidPositions[k]), _SupportRadius, _KernelNormalization);
        }
    }

    return density;
}

void SphAddRigidFactor(FS_VEC position, inout FS_VEC sum, inout float sumSq, inout uint neighborCount)
{
    if (_RigidParticleCount == 0u)
    {
        return;
    }

    FS_IVEC coord = SphCellCoord(position);
    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _RigidCellOffsets[cell];
        uint end = start + _RigidCellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            FS_VEC delta = position - _RigidPositions[k];
            if (length(delta) < _SupportRadius)
            {
                neighborCount += 1u;
            }

            FS_VEC gradient = _RigidMasses[k] *
                              SphKernelGradient(delta, _SupportRadius, _KernelNormalization);
            sum += gradient;
            sumSq += dot(gradient, gradient);
        }
    }
}

// SPlisHSPlasH zeros DFSPH k below ~20 neighbours. A hard cut makes the free
// surface (half a neighbourhood) lose all pressure; a fade from a spray-sized
// count up to a typical interior count keeps the sheet and kills droplets.
float SphNeighborFade(uint count)
{
#if FLUIDSIM_DIMENSIONS == 2
    return saturate(((float)count - 4.0) / 6.0);
#else
    return saturate(((float)count - 8.0) / 14.0);
#endif
}

float SphDensityFloor()
{
    return 0.5 * _RestDensity;
}

float SphSafeInvDensitySq(float density)
{
    float clamped = max(density, SphDensityFloor());
    return 1.0 / max(clamped * clamped, 1e-12);
}

FS_VEC SphClampSpeed(FS_VEC velocity)
{
    const float maxSpeed = 12.0;
    float speed = length(velocity);
    if (speed > maxSpeed)
    {
        return velocity * (maxSpeed / speed);
    }

    return velocity;
}

void SphAddRigidNonPressure(FS_VEC position, FS_VEC velocity, inout FS_VEC acceleration)
{
    if (_RigidParticleCount == 0u)
    {
        return;
    }

    FS_IVEC coord = SphCellCoord(position);
    float supportSqEps = 0.01 * _SupportRadius * _SupportRadius;
    float viscScale = 2.0 * (float(FLUIDSIM_DIMENSIONS) + 2.0) * max(_KinematicViscosity, 1e-3);

    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _RigidCellOffsets[cell];
        uint end = start + _RigidCellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            FS_VEC offset = position - _RigidPositions[k];
            FS_VEC gradient = SphKernelGradient(offset, _SupportRadius, _KernelNormalization);
            float denom = dot(offset, offset) + supportSqEps;
            if (_ImplicitViscosityIterations < 0.5)
            {
                acceleration += viscScale * (_RigidMasses[k] / max(_RestDensity, 1e-12)) *
                                (dot(velocity - _RigidVelocities[k], offset) / denom) * gradient;
            }

            if (_Adhesion > 0.0)
            {
                float distance = length(offset);
                if (distance > 1e-8 && distance < _SupportRadius)
                {
                    acceleration -= _Adhesion * _RigidMasses[k] *
                                    SphAkinciAdhesion(distance, _SupportRadius) *
                                    (offset / distance);
                }
            }
        }
    }
}

float SphRigidDivergence(FS_VEC position, FS_VEC velocity)
{
    if (_RigidParticleCount == 0u)
    {
        return 0.0;
    }

    FS_IVEC coord = SphCellCoord(position);
    float divergence = 0.0;
    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _RigidCellOffsets[cell];
        uint end = start + _RigidCellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            FS_VEC gradient = SphKernelGradient(position - _RigidPositions[k], _SupportRadius, _KernelNormalization);
            divergence += _RigidMasses[k] * dot(velocity - _RigidVelocities[k], gradient);
        }
    }

    return divergence;
}

void SphAddRigidPressure(FS_VEC position, float pressure, float invDensitySq, float restInvDensitySq, inout FS_VEC correction)
{
    if (_RigidParticleCount == 0u)
    {
        return;
    }

    FS_IVEC coord = SphCellCoord(position);
    for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
    {
        FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
        if (!SphCellCoordInBounds(neighborCoord))
        {
            continue;
        }

        uint cell = SphCellIndex(neighborCoord);
        uint start = _RigidCellOffsets[cell];
        uint end = start + _RigidCellCounts[cell];
        for (uint k = start; k < end; ++k)
        {
            FS_VEC gradient = SphKernelGradient(position - _RigidPositions[k], _SupportRadius, _KernelNormalization);
            correction += _RigidMasses[k] * (pressure * invDensitySq + pressure * restInvDensitySq) * gradient;
        }
    }
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
            density += _ParticleMass *
                       SphKernel(length(position - _SortedPositions[k]), _SupportRadius, _KernelNormalization);
        }
    }

    _Densities[particle] = density + SphBoundaryDensity(position) + SphRigidDensity(position);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ComputeFactor(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_IVEC coord = SphCellCoord(position);
    FS_VEC sum = FS_ZERO;
    float sumSq = 0.0;
    uint neighborCount = 0u;

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

            FS_VEC delta = position - _SortedPositions[k];
            if (length(delta) < _SupportRadius)
            {
                neighborCount += 1u;
            }

            FS_VEC gradient = _ParticleMass *
                              SphKernelGradient(delta, _SupportRadius, _KernelNormalization);
            sum += gradient;
            sumSq += dot(gradient, gradient);
        }
    }

    if (_BoundaryParticleCount > 0u)
    {
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
                FS_VEC delta = position - _BoundaryPositions[k];
                if (length(delta) < _SupportRadius)
                {
                    neighborCount += 1u;
                }

                FS_VEC gradient = _BoundaryMasses[k] *
                                  SphKernelGradient(delta, _SupportRadius, _KernelNormalization);
                sum += gradient;
                sumSq += dot(gradient, gradient);
            }
        }
    }

    SphAddRigidFactor(position, sum, sumSq, neighborCount);

    float density = _Densities[particle];
    float denom = dot(sum, sum) + sumSq;

    // Cubic spline ∇W vanishes at r = 0. Two particles that have clustered share
    // a huge density but a near-zero denominator, so k = ρ²/ε becomes a bomb.
    // Thinning the neighbourhood shrinks the denom and drives k into the cap,
    // which is the opposite of what a droplet needs: the last neighbours get
    // the stiffest kick. Fade k out as the count drops (SPlisHSPlasH zeros it
    // below ~20) and still cap the interior value.
    const float minDenom = 1e-6;
    const float maxFactor = 0.02;
    float k = denom > minDenom ? min((density * density) / denom, maxFactor) : 0.0;
    _StiffnessFactors[particle] = k * SphNeighborFade(neighborCount);
}

// Akinci et al. 2013 cohesion spline. Attractive in the outer half of the
// support, mildly repulsive near the origin so the term does not collapse a
// pair onto a single point.
float SphAkinciCohesion(float distance, float support)
{
    if (distance <= 1e-8 || distance >= support)
    {
        return 0.0;
    }

    float h2 = support * support;
    float h3 = h2 * support;
#if FLUIDSIM_DIMENSIONS == 2
    // Two isolated 2D particles already reconstruct a large self-density
    // (mass is ρ0 h~²), so 2ρ0/(ρi+ρj) is much smaller than in 3D. The
    // extra factor keeps an authored γ in the same ballpark for the 2D
    // diagnostic mode.
    float scale = 64.0 / (3.14159265 * h3 * h2 * h2);
#else
    float scale = 32.0 / (3.14159265 * h3 * h3 * h3);
#endif

    float gap = support - distance;
    float gap3 = gap * gap * gap;
    float dist3 = distance * distance * distance;
    if (distance > 0.5 * support)
    {
        return scale * gap3 * dist3;
    }

    return scale * (2.0 * gap3 * dist3 - h3 * h3 / 64.0);
}

float SphAkinciAdhesion(float distance, float support)
{
    if (distance <= 0.5 * support || distance >= support)
    {
        return 0.0;
    }

    float argument = -4.0 * distance * distance / support + 6.0 * distance - 2.0 * support;
    if (argument <= 0.0)
    {
        return 0.0;
    }

    return 0.007 / pow(support, 3.25) * pow(argument, 0.25);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ComputeSurfaceNormals(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_IVEC coord = SphCellCoord(position);
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

            FS_VEC gradient = SphKernelGradient(position - _SortedPositions[k], _SupportRadius, _KernelNormalization);
            normal += (_ParticleMass / max(_Densities[k], SphDensityFloor())) * gradient;
        }
    }

    _SurfaceNormals[particle] = _SupportRadius * normal;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ReorderAngularVelocities(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    _SortedAngularVelocities[particle] = _AngularVelocities[_SortedIndices[particle]];
}

// Bender et al. 2017 difference-curl. In 2D the microrotation is the z component.
FS_VEC SphOmegaCrossGradient(float3 omega, FS_VEC gradient)
{
#if FLUIDSIM_DIMENSIONS == 2
    return float2(-omega.z * gradient.y, omega.z * gradient.x);
#else
    return cross(omega, gradient);
#endif
}

float3 SphVelocityCrossGradient(FS_VEC deltaVelocity, FS_VEC gradient)
{
#if FLUIDSIM_DIMENSIONS == 2
    return float3(0.0, 0.0, deltaVelocity.x * gradient.y - deltaVelocity.y * gradient.x);
#else
    return cross(deltaVelocity, gradient);
#endif
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void PredictVelocity(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_VEC velocity = _SortedVelocities[particle];
    FS_IVEC coord = SphCellCoord(position);
    float supportSqEps = 0.01 * _SupportRadius * _SupportRadius;
    // Authored water viscosity is 1e-6 and does nothing at this scale. A small
    // floor keeps the non-pressure predict from delivering an undamped shock
    // into the Jacobi solvers. SESPH keeps the authored value.
    float viscScale = 2.0 * (float(FLUIDSIM_DIMENSIONS) + 2.0) * max(_KinematicViscosity, 1e-3);
    float3 omega = _SortedAngularVelocities[particle];
    float3 angularAcceleration = float3(0.0, 0.0, 0.0);
    float invDt = 1.0 / max(_DeltaTime, 1e-6);
    float density = max(_Densities[particle], SphDensityFloor());

    FS_VEC acceleration = FS_GRAVITY;
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
            float denom = dot(offset, offset) + supportSqEps;
            if (_ImplicitViscosityIterations < 0.5)
            {
                acceleration += viscScale * (_ParticleMass / max(_Densities[k], SphDensityFloor())) *
                                (dot(velocity - _SortedVelocities[k], offset) / denom) * gradient;
            }

            if (_SurfaceTension > 0.0)
            {
                float distance = length(offset);
                if (distance > 1e-8 && distance < _SupportRadius)
                {
                    FS_VEC rhat = offset / distance;
                    float symmetry = min(2.0 * _RestDensity /
                                     max(_Densities[particle] + _Densities[k], SphDensityFloor()), 2.0);
                    acceleration -= _SurfaceTension * _ParticleMass *
                                    SphAkinciCohesion(distance, _SupportRadius) * symmetry * rhat;
                    acceleration -= _SurfaceTension * (_SurfaceNormals[particle] - _SurfaceNormals[k]);
                }
            }

            if (_Vorticity > 0.0 || _ViscosityOmega > 0.0)
            {
                float3 omegaNeighbor = _SortedAngularVelocities[k];
                float3 omegaDiff = omega - omegaNeighbor;
                float neighborDensity = max(_Densities[k], SphDensityFloor());
                float massOverDensity = _ParticleMass / neighborDensity;

                angularAcceleration -= invDt * _InertiaInverse * _ViscosityOmega * massOverDensity *
                                       omegaDiff * SphKernel(length(offset), _SupportRadius, _KernelNormalization);
                acceleration += (_Vorticity / density) * _ParticleMass * SphOmegaCrossGradient(omegaDiff, gradient);
                angularAcceleration += (_Vorticity / density) * _InertiaInverse * _ParticleMass *
                                       SphVelocityCrossGradient(velocity - _SortedVelocities[k], gradient);
            }
        }
    }

    if (_BoundaryParticleCount > 0u)
    {
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
                float denom = dot(offset, offset) + supportSqEps;
                if (_ImplicitViscosityIterations < 0.5)
                {
                    acceleration += viscScale * (_BoundaryMasses[k] / max(_RestDensity, 1e-12)) *
                                    (dot(velocity, offset) / denom) * gradient;
                }

                if (_Adhesion > 0.0)
                {
                    float distance = length(offset);
                    if (distance > 1e-8 && distance < _SupportRadius)
                    {
                        acceleration -= _Adhesion * _BoundaryMasses[k] *
                                        SphAkinciAdhesion(distance, _SupportRadius) *
                                        (offset / distance);
                    }
                }
            }
        }
    }

    SphAddRigidNonPressure(position, velocity, acceleration);

    if (_Vorticity > 0.0 || _ViscosityOmega > 0.0)
    {
        angularAcceleration -= 2.0 * _InertiaInverse * _Vorticity * omega;
        float3 omegaNew = omega + _DeltaTime * angularAcceleration;
        _SortedAngularVelocities[particle] = omegaNew;
        _AngularVelocities[_SortedIndices[particle]] = omegaNew;
    }

    _PredictedVelocities[particle] = SphClampSpeed(velocity + _DeltaTime * acceleration);
}

float SphVelocityDivergence(uint particle, FS_VEC position, FS_VEC velocity)
{
    FS_IVEC coord = SphCellCoord(position);
    float divergence = 0.0;

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
            FS_VEC gradient = SphKernelGradient(position - _SortedPositions[k], _SupportRadius, _KernelNormalization);
            divergence += _ParticleMass * dot(velocity - _PredictedVelocities[k], gradient);
        }
    }

    if (_BoundaryParticleCount > 0u)
    {
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
                FS_VEC gradient = SphKernelGradient(position - _BoundaryPositions[k], _SupportRadius, _KernelNormalization);
                divergence += _BoundaryMasses[k] * dot(velocity, gradient);
            }
        }
    }

    return divergence + SphRigidDivergence(position, velocity);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void PredictDensity(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_VEC velocity = _PredictedVelocities[particle];
    float predicted = _Densities[particle] + _DeltaTime * SphVelocityDivergence(particle, position, velocity);
    _PredictedDensities[particle] = predicted;

    // Weaker, shorter Jacobi steps: a yellow (slightly over-dense) patch should
    // breathe out, not swap sides with its neighbours in one frame.
    const float relaxation = 0.35;
    float error = min(max(predicted - _RestDensity, 0.0), 0.08 * _RestDensity);
    _Pressures[particle] = relaxation * error * _StiffnessFactors[particle] /
                           max(_DeltaTime * _DeltaTime, 1e-12);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ComputeDivergence(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_VEC velocity = _PredictedVelocities[particle];
    float divergence = SphVelocityDivergence(particle, position, velocity);
    const float relaxation = 0.35;
    float source = min(max(divergence, 0.0), 0.08 * _RestDensity / max(_DeltaTime, 1e-12));
    _Pressures[particle] = relaxation * source * _StiffnessFactors[particle] / max(_DeltaTime, 1e-12);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ApplyPressure(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_IVEC coord = SphCellCoord(position);
    float pressure = _Pressures[particle];
    float invDensitySq = SphSafeInvDensitySq(_Densities[particle]);
    float restInvDensitySq = SphSafeInvDensitySq(_RestDensity);

    FS_VEC correction = FS_ZERO;
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

            FS_VEC gradient = SphKernelGradient(position - _SortedPositions[k], _SupportRadius, _KernelNormalization);
            float neighborInv = SphSafeInvDensitySq(_Densities[k]);
            correction += _ParticleMass * (pressure * invDensitySq + _Pressures[k] * neighborInv) * gradient;
        }
    }

    if (_BoundaryParticleCount > 0u)
    {
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
                FS_VEC gradient = SphKernelGradient(position - _BoundaryPositions[k], _SupportRadius, _KernelNormalization);
                correction += _BoundaryMasses[k] * (pressure * invDensitySq + pressure * restInvDensitySq) * gradient;
            }
        }
    }

    SphAddRigidPressure(position, pressure, invDensitySq, restInvDensitySq, correction);

    FS_VEC deltaV = _DeltaTime * correction;
    // A hard world-speed cap, not only support/Δt: as Δt shrinks the 1/Δt²
    // pressure grows and support/Δt gets looser, which is the opposite of what
    // a settling (yellow) volume needs.
    float maxDelta = min(0.2 * _SupportRadius / max(_DeltaTime, 1e-6), 2.0);
    float speed = length(deltaV);
    if (speed > maxDelta)
    {
        deltaV *= maxDelta / speed;
    }

    _PredictedVelocities[particle] -= deltaV;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void IntegratePositions(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC velocity = SphClampSpeed(_PredictedVelocities[particle]);
    FS_VEC position = _SortedPositions[particle] + _DeltaTime * velocity;
    SphCollideBox(position, velocity);
    velocity = SphClampSpeed(velocity);

    _SortedPositions[particle] = position;
    _PredictedVelocities[particle] = velocity;

    uint source = _SortedIndices[particle];
    _Positions[source] = position;
    _Velocities[source] = velocity;
}

// After the host rebuilds the neighbour grid, predicted velocities live in the
// unsorted buffer and have just been gathered into _SortedVelocities. Copy them
// so the second half of the step can keep using the predicted-velocity kernels.
[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void SeedPredictedVelocities(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    _PredictedVelocities[particle] = _SortedVelocities[particle];
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void FinalizeVelocities(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC velocity = SphClampSpeed(_PredictedVelocities[particle]);
    _SortedVelocities[particle] = velocity;
    _Velocities[_SortedIndices[particle]] = velocity;
    _Speeds[particle] = length(velocity);
}

// Action-reaction onto the rigid samples. Pressure uses the current Jacobi
// κ stored in _Pressures; non-pressure mirrors PredictVelocity's wall terms
// with the rigid particle velocity included.
[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void AccumulateRigidForces(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _RigidParticleCount)
    {
        return;
    }

    FS_VEC position = _RigidPositions[particle];
    FS_VEC velocity = _RigidVelocities[particle];
    float mass = _RigidMasses[particle];
    float3 force = _RigidForces[particle];

    FS_IVEC coord = SphCellCoord(position);
    float restInvDensitySq = SphSafeInvDensitySq(_RestDensity);
    float supportSqEps = 0.01 * _SupportRadius * _SupportRadius;
    float viscScale = 2.0 * (float(FLUIDSIM_DIMENSIONS) + 2.0) * max(_KinematicViscosity, 1e-3);

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
            FS_VEC offset = _SortedPositions[k] - position;
            FS_VEC gradient = SphKernelGradient(offset, _SupportRadius, _KernelNormalization);

            if (_RigidAccumulateMode == 1)
            {
                float pressure = _Pressures[k];
                float invDensitySq = SphSafeInvDensitySq(_Densities[k]);
                force += SphToFloat3(_ParticleMass * mass *
                                     (pressure * invDensitySq + pressure * restInvDensitySq) * gradient);
                continue;
            }

            float denom = dot(offset, offset) + supportSqEps;
            FS_VEC fluidVelocity = _ImplicitViscosityIterations > 0.5
                ? _PredictedVelocities[k]
                : _SortedVelocities[k];
            FS_VEC viscAccel = viscScale * (mass / max(_RestDensity, 1e-12)) *
                               (dot(fluidVelocity - velocity, offset) / denom) * gradient;
            force -= SphToFloat3(_ParticleMass * viscAccel);

            if (_Adhesion > 0.0)
            {
                float distance = length(offset);
                if (distance > 1e-8 && distance < _SupportRadius)
                {
                    force += SphToFloat3(_ParticleMass * _Adhesion * mass *
                                         SphAkinciAdhesion(distance, _SupportRadius) *
                                         (offset / distance));
                }
            }
        }
    }

    _RigidForces[particle] = force;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void SeedViscosityState(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC predicted = _PredictedVelocities[particle];
    _ViscosityVelocities[particle] = predicted;
    _ViscosityRhs[particle] = predicted;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void CommitViscosityIterate(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    _ViscosityVelocities[particle] = _PredictedVelocities[particle];
}

// Weiler 2018 implicit Brookshaw, Picard/Jacobi: v = v* + Δt Visc(v).
// Same pair term as PredictVelocity, so a receding pair is guaranteed to
// feel the operator. Under-relaxed so large ν stays stable.
[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void JacobiViscosity(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_VEC velocity = _ViscosityVelocities[particle];
    FS_VEC target = _ViscosityRhs[particle];
    FS_IVEC coord = SphCellCoord(position);
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
            float denom = dot(offset, offset) + supportSqEps;
            acceleration += viscScale * (_ParticleMass / max(_Densities[k], SphDensityFloor())) *
                            (dot(velocity - _ViscosityVelocities[k], offset) / denom) * gradient;
        }
    }

    if (_BoundaryParticleCount > 0u)
    {
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
                float denom = dot(offset, offset) + supportSqEps;
                acceleration += viscScale * (_BoundaryMasses[k] / max(_RestDensity, 1e-12)) *
                                (dot(velocity, offset) / denom) * gradient;
            }
        }
    }

    if (_RigidParticleCount > 0u)
    {
        for (uint n = 0u; n < FS_NEIGHBOR_CELL_COUNT; ++n)
        {
            FS_IVEC neighborCoord = coord + SphNeighborCellOffset(n);
            if (!SphCellCoordInBounds(neighborCoord))
            {
                continue;
            }

            uint cell = SphCellIndex(neighborCoord);
            uint start = _RigidCellOffsets[cell];
            uint end = start + _RigidCellCounts[cell];
            for (uint k = start; k < end; ++k)
            {
                FS_VEC offset = position - _RigidPositions[k];
                FS_VEC gradient = SphKernelGradient(offset, _SupportRadius, _KernelNormalization);
                float denom = dot(offset, offset) + supportSqEps;
                acceleration += viscScale * (_RigidMasses[k] / max(_RestDensity, 1e-12)) *
                                (dot(velocity - _RigidVelocities[k], offset) / denom) * gradient;
            }
        }
    }

    const float relaxation = 0.5;
    _PredictedVelocities[particle] = lerp(velocity, target + _DeltaTime * acceleration, relaxation);
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ScatterRigidMasses(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _RigidParticleCount)
    {
        return;
    }

    _UnsortedRigidMasses[_RigidSortedIndices[particle]] = _RigidMasses[particle];
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ReorderRigidMasses(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _RigidParticleCount)
    {
        return;
    }

    _RigidMassesOut[particle] = _UnsortedRigidMasses[_RigidSortedIndices[particle]];
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ClearRigidForces(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _RigidParticleCount)
    {
        return;
    }

    _RigidForces[particle] = float3(0.0, 0.0, 0.0);
}

#define FS_RIGID_REDUCE_SIZE 256

groupshared float3 FsRigidForceShared[FS_RIGID_REDUCE_SIZE];
groupshared float3 FsRigidTorqueShared[FS_RIGID_REDUCE_SIZE];

[numthreads(FS_RIGID_REDUCE_SIZE, 1, 1)]
void ReduceRigidForces(uint3 id : SV_DispatchThreadID)
{
    float3 force = float3(0.0, 0.0, 0.0);
    float3 torque = float3(0.0, 0.0, 0.0);
    float3 com = _RigidCenterOfMass.xyz;

    for (uint i = id.x; i < _RigidParticleCount; i += FS_RIGID_REDUCE_SIZE)
    {
        float3 particleForce = _RigidForces[i];
        force += particleForce;
        torque += cross(SphToFloat3(_RigidPositions[i]) - com, particleForce);
    }

    FsRigidForceShared[id.x] = force;
    FsRigidTorqueShared[id.x] = torque;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = FS_RIGID_REDUCE_SIZE / 2u; stride > 0u; stride >>= 1u)
    {
        if (id.x < stride)
        {
            FsRigidForceShared[id.x] += FsRigidForceShared[id.x + stride];
            FsRigidTorqueShared[id.x] += FsRigidTorqueShared[id.x + stride];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (id.x == 0u)
    {
        _RigidReduced[0] = FsRigidForceShared[0];
        _RigidReduced[1] = FsRigidTorqueShared[0];
    }
}

#endif // FLUIDSIM_DFSPH_BODY_INCLUDED
