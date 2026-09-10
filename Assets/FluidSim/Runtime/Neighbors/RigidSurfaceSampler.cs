using System.Collections.Generic;
using FluidSim.Core;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Neighbors
{
    /// <summary>
    /// One-layer Akinci sampling of a box surface in local space. Edges and corners
    /// are emitted once so a later neighbourhood sum does not double-count mass.
    /// </summary>
    public static class RigidSurfaceSampler
    {
        public static float3[] SampleBox(float3 size, float spacing, SimulationDimension dimension)
        {
            float3 half = math.max(size * 0.5f, new float3(spacing * 0.5f));
            float3 min = -half;
            float3 max = half;

            int nx = AxisCount(max.x - min.x, spacing);
            int ny = AxisCount(max.y - min.y, spacing);
            var points = new List<float3>();

            if (dimension == SimulationDimension.Two)
            {
                for (int i = 0; i < nx; i++)
                {
                    float x = AxisValue(min.x, max.x, i, nx);
                    points.Add(new float3(x, min.y, 0f));
                    points.Add(new float3(x, max.y, 0f));
                }

                for (int j = 1; j < ny - 1; j++)
                {
                    float y = AxisValue(min.y, max.y, j, ny);
                    points.Add(new float3(min.x, y, 0f));
                    points.Add(new float3(max.x, y, 0f));
                }

                return points.ToArray();
            }

            int nz = AxisCount(max.z - min.z, spacing);

            for (int i = 0; i < nx; i++)
            {
                float x = AxisValue(min.x, max.x, i, nx);
                for (int k = 0; k < nz; k++)
                {
                    float z = AxisValue(min.z, max.z, k, nz);
                    points.Add(new float3(x, min.y, z));
                    points.Add(new float3(x, max.y, z));
                }
            }

            for (int j = 1; j < ny - 1; j++)
            {
                float y = AxisValue(min.y, max.y, j, ny);
                for (int k = 0; k < nz; k++)
                {
                    float z = AxisValue(min.z, max.z, k, nz);
                    points.Add(new float3(min.x, y, z));
                    points.Add(new float3(max.x, y, z));
                }
            }

            for (int i = 1; i < nx - 1; i++)
            {
                float x = AxisValue(min.x, max.x, i, nx);
                for (int j = 1; j < ny - 1; j++)
                {
                    float y = AxisValue(min.y, max.y, j, ny);
                    points.Add(new float3(x, y, min.z));
                    points.Add(new float3(x, y, max.z));
                }
            }

            return points.ToArray();
        }

        static int AxisCount(float length, float spacing)
        {
            return math.max(2, (int)math.floor(length / math.max(spacing, 1e-6f)) + 1);
        }

        static float AxisValue(float min, float max, int index, int count)
        {
            return math.lerp(min, max, index / (float)(count - 1));
        }
    }
}
