#ifndef FLUIDSIM_DENSITY_VOLUME_BODY_INCLUDED
#define FLUIDSIM_DENSITY_VOLUME_BODY_INCLUDED

#include "SphGrid.hlsl"
#include "SphKernel.hlsl"

// Voxelise SPH density into a 3D texture the way SebLague/Fluid-Sim
// UpdateDensityTexture does: each voxel is CalculateDensitiesAtPoint at
// its world position. The neighbour grid is already built; we only read it.

#define FS_DENSITY_GROUP 8

StructuredBuffer<FS_VEC> _SortedPositions;
StructuredBuffer<uint> _CellOffsets;
StructuredBuffer<uint> _CellCounts;

RWTexture3D<float> DensityMap;
float4 _DensityMapSize;

float _SupportRadius;
float _KernelNormalization;
float _ParticleMass;
float4 _DomainCenter;
float4 _DomainSize;

float3 VoxelWorldPosition(uint3 id)
{
    float3 mapSize = max(_DensityMapSize.xyz - 1.0, 1.0);
    float3 uvw = float3(id) / mapSize;
    float3 local = (uvw - 0.5) * _DomainSize.xyz;
#if FLUIDSIM_DIMENSIONS == 2
    return float3(_DomainCenter.xy + local.xy, _DomainCenter.z);
#else
    return _DomainCenter.xyz + local;
#endif
}

float SphDensityAt(float3 worldPos)
{
#if FLUIDSIM_DIMENSIONS == 2
    FS_VEC position = worldPos.xy;
#else
    FS_VEC position = worldPos;
#endif

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

    return density;
}

[numthreads(FS_DENSITY_GROUP, FS_DENSITY_GROUP, FS_DENSITY_GROUP)]
void UpdateDensityTexture(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)_DensityMapSize.x || id.y >= (uint)_DensityMapSize.y ||
        id.z >= (uint)_DensityMapSize.z)
    {
        return;
    }

    DensityMap[id] = SphDensityAt(VoxelWorldPosition(id));
}

#endif
