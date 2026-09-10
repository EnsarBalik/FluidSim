#ifndef FLUIDSIM_PARTICLE_BILLBOARD_INCLUDED
#define FLUIDSIM_PARTICLE_BILLBOARD_INCLUDED

// Camera-facing particle quads addressed by SV_VertexID. Shared by the debug discs
// and the screen-space depth / thickness impostors so the corner winding cannot drift.

float _ParticleRadius;
float _PlaneDepth;

#if defined(FLUIDSIM_POSITIONS_3D)
    StructuredBuffer<float3> _Positions;

    float3 LoadParticlePosition(uint particle)
    {
        return _Positions[particle];
    }
#else
    StructuredBuffer<float2> _Positions;

    float3 LoadParticlePosition(uint particle)
    {
        return float3(_Positions[particle], _PlaneDepth);
    }
#endif

float2 ParticleCornerOffset(uint corner)
{
    uint index = corner;
    if (index == 3u)
    {
        index = 2u;
    }
    else if (index == 4u)
    {
        index = 1u;
    }
    else if (index == 5u)
    {
        index = 3u;
    }

    float x = (index & 2u) != 0u ? 1.0 : -1.0;
    float y = (index & 1u) != 0u ? 1.0 : -1.0;
    return float2(x, y);
}

#endif
