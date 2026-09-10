#ifndef FLUIDSIM_FLOOR_TILES_INCLUDED
#define FLUIDSIM_FLOOR_TILES_INCLUDED

// SebLague tiled floor albedo. Shared by the Game-view mesh and by planar
// reflections so a bounce that misses the camera still shows the same tiles.

float4 _TileCol1;
float4 _TileCol2;
float4 _TileCol3;
float4 _TileCol4;
float3 _TileColVariation;
float _TileScale;
float _TileDarkOffset;
float3 _TileOrigin;

float FluidFloorModulo(float x, float y)
{
    return x - y * floor(x / y);
}

float3 FluidFloorRgbToHsv(float3 rgb)
{
    float4 k = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = rgb.g < rgb.b ? float4(rgb.bg, k.wz) : float4(rgb.gb, k.xy);
    float4 q = rgb.r < p.x ? float4(p.xyw, rgb.r) : float4(rgb.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 FluidFloorHsvToRgb(float3 hsv)
{
    float4 k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(hsv.xxx + k.xyz) * 6.0 - k.www);
    return hsv.z * lerp(k.xxx, saturate(p - k.xxx), hsv.y);
}

float3 FluidFloorTweakHsv(float3 colRgb, float3 shift)
{
    return saturate(FluidFloorHsvToRgb(FluidFloorRgbToHsv(colRgb) + shift));
}

uint FluidFloorHashInt2(int2 v)
{
    return (uint)(v.x * 5023 + v.y * 96456);
}

uint FluidFloorNextRandom(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint result = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (result >> 22u) ^ result;
}

float FluidFloorRandomSNorm(inout uint state)
{
    return FluidFloorNextRandom(state) / 4294967295.0 * 2.0 - 1.0;
}

int FluidFloorQuadrant(float x, float z)
{
    if (x < 0.0 && z < 0.0)
    {
        return 3;
    }

    if (x < 0.0)
    {
        return 1;
    }

    if (z < 0.0)
    {
        return 4;
    }

    return 2;
}

float3 FluidFloorAlbedo(float3 worldPos)
{
    float3 local = worldPos - _TileOrigin;
    float3 tileCol = local.x < 0.0 ? _TileCol1.rgb : _TileCol2.rgb;
    if (local.z < 0.0)
    {
        tileCol = local.x < 0.0 ? _TileCol3.rgb : _TileCol4.rgb;
    }

    int2 tileCoord = int2(floor(worldPos.xz * max(_TileScale, 1e-4)));
    bool isDarkTile = FluidFloorModulo((float)tileCoord.x, 2.0) ==
                      FluidFloorModulo((float)tileCoord.y, 2.0);
    tileCol = FluidFloorTweakHsv(tileCol, float3(0.0, 0.0, _TileDarkOffset * isDarkTile));

    uint rng = FluidFloorHashInt2(tileCoord);
    float3 jitter = float3(
        FluidFloorRandomSNorm(rng), FluidFloorRandomSNorm(rng), FluidFloorRandomSNorm(rng)) *
                    _TileColVariation;
    return FluidFloorTweakHsv(tileCol, jitter);
}

#endif
