using System;
using FluidSim.Core;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Neighbors
{
    /// <summary>
    /// Uniform grid neighbour search built by counting sort on the GPU.
    ///
    /// Five dispatches per rebuild: clear the cell counters, count particles per cell with
    /// one atomic each, exclusive-scan the counters into per-cell start offsets, scatter the
    /// particle indices into their cell's slice, then gather particle data into that order.
    /// The result is O(n) and leaves the particle data laid out so that threads processing
    /// adjacent particles read overlapping memory.
    /// </summary>
    public sealed class NeighborGrid : IDisposable
    {
        // Must match FS_PARTICLE_GROUP_SIZE and FS_CELL_GROUP_SIZE in NeighborGridBody.hlsl.
        const int ParticleGroupSize = 64;
        const int CellGroupSize = 64;

        static readonly int PositionsId = Shader.PropertyToID("_Positions");
        static readonly int VelocitiesId = Shader.PropertyToID("_Velocities");
        static readonly int SortedPositionsId = Shader.PropertyToID("_SortedPositions");
        static readonly int SortedVelocitiesId = Shader.PropertyToID("_SortedVelocities");
        static readonly int CellCountsId = Shader.PropertyToID("_CellCounts");
        static readonly int CellOffsetsId = Shader.PropertyToID("_CellOffsets");
        static readonly int CellCursorsId = Shader.PropertyToID("_CellCursors");
        static readonly int ParticleCellIndicesId = Shader.PropertyToID("_ParticleCellIndices");
        static readonly int SortedIndicesId = Shader.PropertyToID("_SortedIndices");
        static readonly int NeighborCountsId = Shader.PropertyToID("_NeighborCounts");
        static readonly int ReferenceNeighborCountsId = Shader.PropertyToID("_ReferenceNeighborCounts");
        static readonly int DomainMinId = Shader.PropertyToID("_DomainMin");
        static readonly int GridResolutionId = Shader.PropertyToID("_GridResolution");
        static readonly int CellSizeId = Shader.PropertyToID("_CellSize");
        static readonly int InverseCellSizeId = Shader.PropertyToID("_InverseCellSize");
        static readonly int CellCountId = Shader.PropertyToID("_CellCount");
        static readonly int ParticleCountId = Shader.PropertyToID("_ParticleCount");
        static readonly int SupportRadiusId = Shader.PropertyToID("_SupportRadius");

        readonly ComputeShader shader;
        readonly int clearCellsKernel;
        readonly int countCellsKernel;
        readonly int prefixSumKernel;
        readonly int scatterKernel;
        readonly int reorderKernel;
        readonly int countNeighborsKernel;
        readonly int countReferenceKernel;

        public int Capacity { get; }

        public int CellCount { get; }

        public int3 Resolution { get; }

        public ComputeBuffer CellCounts { get; private set; }

        public ComputeBuffer CellOffsets { get; private set; }

        public ComputeBuffer CellCursors { get; private set; }

        public ComputeBuffer ParticleCellIndices { get; private set; }

        public ComputeBuffer SortedIndices { get; private set; }

        /// <summary>Neighbours within the support radius, counting the particle itself.</summary>
        public ComputeBuffer NeighborCounts { get; private set; }

        /// <summary>Quadratic ground truth for <see cref="NeighborCounts"/>.</summary>
        public ComputeBuffer ReferenceNeighborCounts { get; private set; }

        public NeighborGrid(ComputeShader shader, FluidParameters parameters, int capacity)
        {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));

            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }

            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be positive.");
            }

            Capacity = capacity;
            CellCount = parameters.CellCount;
            Resolution = parameters.GridResolution;

            clearCellsKernel = shader.FindKernel("ClearCells");
            countCellsKernel = shader.FindKernel("CountCells");
            prefixSumKernel = shader.FindKernel("PrefixSumCells");
            scatterKernel = shader.FindKernel("ScatterParticles");
            reorderKernel = shader.FindKernel("ReorderParticles");
            countNeighborsKernel = shader.FindKernel("CountNeighbors");
            countReferenceKernel = shader.FindKernel("CountNeighborsReference");

            CellCounts = new ComputeBuffer(CellCount, sizeof(uint));
            CellOffsets = new ComputeBuffer(CellCount, sizeof(uint));
            CellCursors = new ComputeBuffer(CellCount, sizeof(uint));
            ParticleCellIndices = new ComputeBuffer(capacity, sizeof(uint));
            SortedIndices = new ComputeBuffer(capacity, sizeof(uint));
            NeighborCounts = new ComputeBuffer(capacity, sizeof(float));
            ReferenceNeighborCounts = new ComputeBuffer(capacity, sizeof(float));
        }

        /// <summary>
        /// Rebuilds the grid from scratch. Doing so every step rather than updating it
        /// incrementally costs the same regardless of how far the particles moved.
        /// </summary>
        public void Build(ParticleSet particles, FluidParameters parameters)
        {
            if (particles == null)
            {
                throw new ArgumentNullException(nameof(particles));
            }

            RequireMatchingLayout(particles, parameters);

            if (particles.Count == 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count);

            shader.SetBuffer(clearCellsKernel, CellCountsId, CellCounts);
            shader.SetBuffer(clearCellsKernel, CellCursorsId, CellCursors);
            DispatchOver(clearCellsKernel, CellCount, CellGroupSize);

            shader.SetBuffer(countCellsKernel, PositionsId, particles.Positions);
            shader.SetBuffer(countCellsKernel, CellCountsId, CellCounts);
            shader.SetBuffer(countCellsKernel, ParticleCellIndicesId, ParticleCellIndices);
            DispatchOver(countCellsKernel, particles.Count, ParticleGroupSize);

            shader.SetBuffer(prefixSumKernel, CellCountsId, CellCounts);
            shader.SetBuffer(prefixSumKernel, CellOffsetsId, CellOffsets);
            shader.Dispatch(prefixSumKernel, 1, 1, 1);

            shader.SetBuffer(scatterKernel, CellCursorsId, CellCursors);
            shader.SetBuffer(scatterKernel, CellOffsetsId, CellOffsets);
            shader.SetBuffer(scatterKernel, ParticleCellIndicesId, ParticleCellIndices);
            shader.SetBuffer(scatterKernel, SortedIndicesId, SortedIndices);
            DispatchOver(scatterKernel, particles.Count, ParticleGroupSize);

            shader.SetBuffer(reorderKernel, PositionsId, particles.Positions);
            shader.SetBuffer(reorderKernel, VelocitiesId, particles.Velocities);
            shader.SetBuffer(reorderKernel, SortedIndicesId, SortedIndices);
            shader.SetBuffer(reorderKernel, SortedPositionsId, particles.SortedPositions);
            shader.SetBuffer(reorderKernel, SortedVelocitiesId, particles.SortedVelocities);
            DispatchOver(reorderKernel, particles.Count, ParticleGroupSize);
        }

        /// <summary>Counts neighbours using the grid. Requires a preceding <see cref="Build"/>.</summary>
        public void CountNeighbors(ParticleSet particles, FluidParameters parameters)
        {
            RequireMatchingLayout(particles, parameters);

            if (particles.Count == 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count);

            shader.SetBuffer(countNeighborsKernel, SortedPositionsId, particles.SortedPositions);
            shader.SetBuffer(countNeighborsKernel, CellOffsetsId, CellOffsets);
            shader.SetBuffer(countNeighborsKernel, CellCountsId, CellCounts);
            shader.SetBuffer(countNeighborsKernel, NeighborCountsId, NeighborCounts);
            DispatchOver(countNeighborsKernel, particles.Count, ParticleGroupSize);
        }

        /// <summary>
        /// Counts neighbours by brute force for validation. Quadratic, so only for tests and
        /// deliberate self-checks.
        /// </summary>
        public void CountNeighborsReference(ParticleSet particles, FluidParameters parameters)
        {
            RequireMatchingLayout(particles, parameters);

            if (particles.Count == 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count);

            shader.SetBuffer(countReferenceKernel, SortedPositionsId, particles.SortedPositions);
            shader.SetBuffer(countReferenceKernel, ReferenceNeighborCountsId, ReferenceNeighborCounts);
            DispatchOver(countReferenceKernel, particles.Count, ParticleGroupSize);
        }

        public uint[] ReadCellCounts()
        {
            var data = new uint[CellCount];
            CellCounts.GetData(data);
            return data;
        }

        public uint[] ReadSortedIndices(int count)
        {
            var data = new uint[count];
            if (count > 0)
            {
                SortedIndices.GetData(data, 0, 0, count);
            }

            return data;
        }

        public uint[] ReadParticleCellIndices(int count)
        {
            var data = new uint[count];
            if (count > 0)
            {
                ParticleCellIndices.GetData(data, 0, 0, count);
            }

            return data;
        }

        public float[] ReadNeighborCounts(int count)
        {
            return Read(NeighborCounts, count);
        }

        public float[] ReadReferenceNeighborCounts(int count)
        {
            return Read(ReferenceNeighborCounts, count);
        }

        public void Dispose()
        {
            CellCounts?.Release();
            CellOffsets?.Release();
            CellCursors?.Release();
            ParticleCellIndices?.Release();
            SortedIndices?.Release();
            NeighborCounts?.Release();
            ReferenceNeighborCounts?.Release();

            CellCounts = null;
            CellOffsets = null;
            CellCursors = null;
            ParticleCellIndices = null;
            SortedIndices = null;
            NeighborCounts = null;
            ReferenceNeighborCounts = null;
        }

        static float[] Read(ComputeBuffer buffer, int count)
        {
            var data = new float[count];
            if (count > 0)
            {
                buffer.GetData(data, 0, 0, count);
            }

            return data;
        }

        void SetUniforms(FluidParameters parameters, int particleCount)
        {
            Bounds domain = parameters.Domain;
            int3 resolution = parameters.GridResolution;

            shader.SetVector(DomainMinId, new Vector4(domain.min.x, domain.min.y, domain.min.z, 0f));
            shader.SetVector(GridResolutionId, new Vector4(resolution.x, resolution.y, resolution.z, 0f));
            shader.SetFloat(CellSizeId, parameters.CellSize);
            shader.SetFloat(InverseCellSizeId, 1f / parameters.CellSize);
            shader.SetInt(CellCountId, CellCount);
            shader.SetInt(ParticleCountId, particleCount);
            shader.SetFloat(SupportRadiusId, parameters.SupportRadius);
        }

        void DispatchOver(int kernel, int elementCount, int groupSize)
        {
            if (elementCount <= 0)
            {
                return;
            }

            shader.Dispatch(kernel, (elementCount + groupSize - 1) / groupSize, 1, 1);
        }

        /// <summary>
        /// The buffers are sized once from the parameters, so editing the domain or the
        /// particle radius while the grid is alive would silently address the wrong cells.
        /// </summary>
        void RequireMatchingLayout(ParticleSet particles, FluidParameters parameters)
        {
            if (parameters.CellCount != CellCount)
            {
                throw new InvalidOperationException(
                    $"The grid was allocated for {CellCount} cells but the parameters now " +
                    $"describe {parameters.CellCount}. Reallocate the simulation after changing " +
                    "the domain or the particle radius.");
            }

            if (particles.Count > Capacity)
            {
                throw new InvalidOperationException(
                    $"{particles.Count} particles exceed the grid capacity of {Capacity}.");
            }
        }
    }
}
