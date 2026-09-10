using System.Collections.Generic;
using FluidSim.Core;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Neighbors
{
    /// <summary>
    /// One-layer sampling of the simulation domain, Akinci-style. Particles sit on the
    /// faces of the box so a fluid particle that approaches the wall still has a full
    /// kernel support on that side. Edges and corners are emitted once.
    /// </summary>
    public static class DomainBoundarySampler
    {
        public static float3[] Sample(FluidParameters parameters)
        {
            Bounds domain = parameters.Domain;
            float spacing = parameters.ParticleSize;
            float3 min = domain.min;
            float3 max = domain.max;

            int nx = AxisCount(max.x - min.x, spacing);
            int ny = AxisCount(max.y - min.y, spacing);

            var points = new List<float3>();

            if (parameters.Dimension == SimulationDimension.Two)
            {
                float z = domain.center.z;
                for (int i = 0; i < nx; i++)
                {
                    float x = AxisValue(min.x, max.x, i, nx);
                    points.Add(new float3(x, min.y, z));
                    points.Add(new float3(x, max.y, z));
                }

                for (int j = 1; j < ny - 1; j++)
                {
                    float y = AxisValue(min.y, max.y, j, ny);
                    points.Add(new float3(min.x, y, z));
                    points.Add(new float3(max.x, y, z));
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
            return math.max(2, (int)math.floor(length / spacing) + 1);
        }

        static float AxisValue(float min, float max, int index, int count)
        {
            return math.lerp(min, max, index / (float)(count - 1));
        }
    }
}
