using System;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Core
{
    /// <summary>
    /// Owns the per-particle GPU buffers in structure-of-arrays layout: one buffer per
    /// attribute rather than one buffer of structs, so each pass only reads the lanes it
    /// needs and neighbouring threads touch adjacent addresses.
    ///
    /// Every attribute exists twice. The unsorted buffers are the authoritative state that
    /// the solver integrates; the sorted buffers are the cell-ordered copies the neighbour
    /// grid produces, and are what the SPH passes actually read.
    /// </summary>
    public sealed class ParticleSet : IDisposable
    {
        public int Dimensions { get; }

        public int Capacity { get; }

        public int Count { get; private set; }

        public ComputeBuffer Positions { get; private set; }

        public ComputeBuffer Velocities { get; private set; }

        public ComputeBuffer SortedPositions { get; private set; }

        public ComputeBuffer SortedVelocities { get; private set; }

        public ParticleSet(int capacity, int dimensions)
        {
            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be positive.");
            }

            if (dimensions != 2 && dimensions != 3)
            {
                throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Only 2 and 3 dimensions exist.");
            }

            Capacity = capacity;
            Dimensions = dimensions;

            int stride = sizeof(float) * dimensions;
            Positions = new ComputeBuffer(capacity, stride);
            Velocities = new ComputeBuffer(capacity, stride);
            SortedPositions = new ComputeBuffer(capacity, stride);
            SortedVelocities = new ComputeBuffer(capacity, stride);
        }

        /// <summary>Uploads an initial configuration at rest and sets the active count.</summary>
        public void Fill(float2[] positions)
        {
            if (positions == null)
            {
                throw new ArgumentNullException(nameof(positions));
            }

            RequireDimensions(2);
            RequireCapacity(positions.Length);

            Count = positions.Length;
            Positions.SetData(positions, 0, 0, Count);
            Velocities.SetData(new float2[Count], 0, 0, Count);
        }

        /// <inheritdoc cref="Fill(float2[])"/>
        public void Fill(float3[] positions)
        {
            if (positions == null)
            {
                throw new ArgumentNullException(nameof(positions));
            }

            RequireDimensions(3);
            RequireCapacity(positions.Length);

            Count = positions.Length;
            Positions.SetData(positions, 0, 0, Count);
            Velocities.SetData(new float3[Count], 0, 0, Count);
        }

        /// <summary>Reads the cell-ordered positions back for inspection. Stalls the GPU.</summary>
        public float2[] ReadSortedPositions2D()
        {
            RequireDimensions(2);

            var data = new float2[Count];
            if (Count > 0)
            {
                SortedPositions.GetData(data, 0, 0, Count);
            }

            return data;
        }

        /// <inheritdoc cref="ReadSortedPositions2D"/>
        public float3[] ReadSortedPositions3D()
        {
            RequireDimensions(3);

            var data = new float3[Count];
            if (Count > 0)
            {
                SortedPositions.GetData(data, 0, 0, Count);
            }

            return data;
        }

        public void Dispose()
        {
            Positions?.Release();
            Velocities?.Release();
            SortedPositions?.Release();
            SortedVelocities?.Release();

            Positions = null;
            Velocities = null;
            SortedPositions = null;
            SortedVelocities = null;
            Count = 0;
        }

        void RequireDimensions(int expected)
        {
            if (Dimensions != expected)
            {
                throw new InvalidOperationException(
                    $"This particle set is {Dimensions}D; the call expects {expected}D data.");
            }
        }

        void RequireCapacity(int required)
        {
            if (required > Capacity)
            {
                throw new ArgumentException(
                    $"{required} particles exceed the allocated capacity of {Capacity}.");
            }
        }
    }
}
