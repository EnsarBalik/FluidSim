#ifndef FLUIDSIM_NEIGHBOR_GRID_BODY_INCLUDED
#define FLUIDSIM_NEIGHBOR_GRID_BODY_INCLUDED

#include "SphGrid.hlsl"

// Neighbour search by counting sort, following the "build the grid by sorting" advice in
// Green's CUDA particles note but replacing the radix sort with a bucket sort, since the
// domain is bounded and the bucket count is therefore known up front.
//
//   ClearCells       zero the per-cell counters
//   CountCells       one atomic increment per particle
//   PrefixSumCells   exclusive scan over the counters, giving each cell its start offset
//   ScatterParticles one atomic bump per particle, writing it into its cell's slice
//   ReorderParticles gather particle data into sorted order for memory coherence
//
// The whole thing is O(n) and takes five dispatches. The scan runs on a single work group
// that loops over the cells in chunks, which keeps it to one dispatch with no group-sum
// bookkeeping and imposes no upper bound on the cell count.

#define FS_PARTICLE_GROUP_SIZE 64
#define FS_CELL_GROUP_SIZE 64
#define FS_SCAN_GROUP_SIZE 512

// The authoritative particle state. Read-only here so these stay in SRV slots.
StructuredBuffer<FS_VEC> _Positions;
StructuredBuffer<FS_VEC> _Velocities;

// Particle data in cell order. Everything downstream of the grid, including the solver,
// reads these. CountNeighbors uses the same name so a missing declaration cannot compile.
RWStructuredBuffer<FS_VEC> _SortedPositions;
RWStructuredBuffer<FS_VEC> _SortedVelocities;

RWStructuredBuffer<uint> _CellCounts;
RWStructuredBuffer<uint> _CellOffsets;
RWStructuredBuffer<uint> _CellCursors;
RWStructuredBuffer<uint> _ParticleCellIndices;
RWStructuredBuffer<uint> _SortedIndices;

// Stored as float rather than uint so the debug renderer can consume any per-particle
// scalar through one code path. Each thread owns its own slot, so no atomics are needed.
RWStructuredBuffer<float> _NeighborCounts;
RWStructuredBuffer<float> _ReferenceNeighborCounts;

float _SupportRadius;

[numthreads(FS_CELL_GROUP_SIZE, 1, 1)]
void ClearCells(uint3 id : SV_DispatchThreadID)
{
    uint cell = id.x;
    if (cell >= _CellCount)
    {
        return;
    }

    _CellCounts[cell] = 0u;
    _CellCursors[cell] = 0u;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void CountCells(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    uint cell = SphCellIndex(SphCellCoord(_Positions[particle]));

    // Cached so the scatter pass does not have to redo the position-to-cell mapping.
    _ParticleCellIndices[particle] = cell;

    InterlockedAdd(_CellCounts[cell], 1u);
}

groupshared uint gScanBuffer[FS_SCAN_GROUP_SIZE];

[numthreads(FS_SCAN_GROUP_SIZE, 1, 1)]
void PrefixSumCells(uint3 groupThreadId : SV_GroupThreadID)
{
    uint lane = groupThreadId.x;
    uint runningTotal = 0u;

    for (uint base = 0u; base < _CellCount; base += FS_SCAN_GROUP_SIZE)
    {
        uint index = base + lane;
        uint value = (index < _CellCount) ? _CellCounts[index] : 0u;

        gScanBuffer[lane] = value;
        GroupMemoryBarrierWithGroupSync();

        // Inclusive Hillis-Steele scan. Every step is read, sync, write, sync so that no
        // lane can overwrite a value another lane has not read yet.
        for (uint offset = 1u; offset < FS_SCAN_GROUP_SIZE; offset <<= 1)
        {
            uint addend = (lane >= offset) ? gScanBuffer[lane - offset] : 0u;
            GroupMemoryBarrierWithGroupSync();
            gScanBuffer[lane] += addend;
            GroupMemoryBarrierWithGroupSync();
        }

        if (index < _CellCount)
        {
            // Exclusive result is the inclusive one minus the element's own contribution,
            // shifted by everything the previous chunks already accounted for.
            _CellOffsets[index] = runningTotal + gScanBuffer[lane] - value;
        }

        uint chunkTotal = gScanBuffer[FS_SCAN_GROUP_SIZE - 1];
        GroupMemoryBarrierWithGroupSync();
        runningTotal += chunkTotal;
    }
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ScatterParticles(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    uint cell = _ParticleCellIndices[particle];

    uint slot;
    InterlockedAdd(_CellCursors[cell], 1u, slot);

    _SortedIndices[_CellOffsets[cell] + slot] = particle;
}

[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void ReorderParticles(uint3 id : SV_DispatchThreadID)
{
    uint sorted = id.x;
    if (sorted >= _ParticleCount)
    {
        return;
    }

    uint source = _SortedIndices[sorted];
    _SortedPositions[sorted] = _Positions[source];
    _SortedVelocities[sorted] = _Velocities[source];
}

// Counts particles within the support radius, including the particle itself, so the result
// is directly comparable to the continuum estimate in FluidParameters.
[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void CountNeighbors(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    FS_IVEC coord = SphCellCoord(position);
    float supportRadiusSq = _SupportRadius * _SupportRadius;

    uint count = 0u;
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
            if (dot(offset, offset) < supportRadiusSq)
            {
                ++count;
            }
        }
    }

    _NeighborCounts[particle] = (float)count;
}

// Ground truth for CountNeighbors. Quadratic and therefore only ever used by tests and
// by the runtime self-check, but it is the only way to be sure the grid is not silently
// dropping neighbours.
[numthreads(FS_PARTICLE_GROUP_SIZE, 1, 1)]
void CountNeighborsReference(uint3 id : SV_DispatchThreadID)
{
    uint particle = id.x;
    if (particle >= _ParticleCount)
    {
        return;
    }

    FS_VEC position = _SortedPositions[particle];
    float supportRadiusSq = _SupportRadius * _SupportRadius;

    uint count = 0u;
    for (uint other = 0u; other < _ParticleCount; ++other)
    {
        FS_VEC offset = position - _SortedPositions[other];
        if (dot(offset, offset) < supportRadiusSq)
        {
            ++count;
        }
    }

    _ReferenceNeighborCounts[particle] = (float)count;
}

#endif // FLUIDSIM_NEIGHBOR_GRID_BODY_INCLUDED
