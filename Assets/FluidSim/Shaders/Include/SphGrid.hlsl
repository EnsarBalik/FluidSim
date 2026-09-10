#ifndef FLUIDSIM_SPH_GRID_INCLUDED
#define FLUIDSIM_SPH_GRID_INCLUDED

#include "SphTypes.hlsl"

// Uniform grid over a bounded domain, with cells exactly as wide as the kernel support
// radius. That sizing is what makes a one-ring query sufficient: every particle within
// the support radius of x_i necessarily lies in the cell of x_i or one of its neighbours.
//
// Resolution is passed as a float4 rather than an int4 on purpose. Integer vector uniforms
// have historically been fussy about constant-buffer packing across graphics APIs, and
// grid resolutions are small enough to be exactly representable as floats.

float4 _DomainMin;
float4 _GridResolution;
float _CellSize;
float _InverseCellSize;
uint _CellCount;
uint _ParticleCount;

#if FLUIDSIM_DIMENSIONS == 2
    #define FS_DOMAIN_MIN (_DomainMin.xy)
    #define FS_GRID_RESOLUTION ((int2)_GridResolution.xy)
#else
    #define FS_DOMAIN_MIN (_DomainMin.xyz)
    #define FS_GRID_RESOLUTION ((int3)_GridResolution.xyz)
#endif

// Clamped so that a particle which has escaped the domain still lands on a valid cell
// instead of corrupting memory. Escaping particles are a boundary-handling problem, not
// something the grid should have an opinion about.
FS_IVEC SphCellCoord(FS_VEC position)
{
    FS_IVEC coord = (FS_IVEC)floor((position - FS_DOMAIN_MIN) * _InverseCellSize);
    return clamp(coord, (FS_IVEC)0, FS_GRID_RESOLUTION - 1);
}

uint SphCellIndex(FS_IVEC coord)
{
#if FLUIDSIM_DIMENSIONS == 2
    return (uint)(coord.x + FS_GRID_RESOLUTION.x * coord.y);
#else
    int3 resolution = FS_GRID_RESOLUTION;
    return (uint)(coord.x + resolution.x * (coord.y + resolution.y * coord.z));
#endif
}

// Decodes a flat one-ring index into a cell offset in [-1, 1]^d, so neighbour iteration
// is a single loop that reads identically in 2D and 3D.
FS_IVEC SphNeighborCellOffset(uint index)
{
#if FLUIDSIM_DIMENSIONS == 2
    return int2((int)(index % 3u), (int)(index / 3u)) - 1;
#else
    return int3((int)(index % 3u), (int)((index / 3u) % 3u), (int)(index / 9u)) - 1;
#endif
}

bool SphCellCoordInBounds(FS_IVEC coord)
{
    return !any(coord < 0) && !any(coord >= FS_GRID_RESOLUTION);
}

#endif // FLUIDSIM_SPH_GRID_INCLUDED
