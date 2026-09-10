#ifndef FLUIDSIM_SPH_TYPES_INCLUDED
#define FLUIDSIM_SPH_TYPES_INCLUDED

// Dimension abstraction for the solver.
//
// Every .compute file defines FLUIDSIM_DIMENSIONS before including anything else and
// then pulls in a shared body. That way the 2D and 3D solvers are the same source with
// a different define, and 2D keeps float2 positions instead of paying for a wasted
// component in the hot neighbour loops.

#ifndef FLUIDSIM_DIMENSIONS
    #define FLUIDSIM_DIMENSIONS 2
#endif

#if FLUIDSIM_DIMENSIONS == 2

    #define FS_VEC   float2
    #define FS_IVEC  int2
    #define FS_ZERO  float2(0.0, 0.0)

    // One-ring of cells around the cell containing a particle: 3x3 in 2D, 3x3x3 in 3D.
    #define FS_NEIGHBOR_CELL_COUNT 9

#elif FLUIDSIM_DIMENSIONS == 3

    #define FS_VEC   float3
    #define FS_IVEC  int3
    #define FS_ZERO  float3(0.0, 0.0, 0.0)

    #define FS_NEIGHBOR_CELL_COUNT 27

#else
    #error "FLUIDSIM_DIMENSIONS must be 2 or 3."
#endif

#endif // FLUIDSIM_SPH_TYPES_INCLUDED
