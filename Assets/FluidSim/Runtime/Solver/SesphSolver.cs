using System;
using FluidSim.Core;
using FluidSim.Neighbors;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Solver
{
    /// <summary>
    /// Weakly compressible SPH (state-equation) solver. Matches Algorithm 1 of the
    /// 2019 tutorial: density from the kernel sum, pressure from p = k (ρ/ρ0 - 1)
    /// clamped at zero, symmetric pressure gradient, Brookshaw viscosity, symplectic Euler.
    ///
    /// Walls are Akinci one-layer particles with pressure mirroring. The kinematic
    /// box clamp remains as a leak safety net; it is not the physical boundary.
    /// </summary>
    public sealed class SesphSolver : IDisposable
    {
        const int ParticleGroupSize = 64;

        static readonly int SortedPositionsId = Shader.PropertyToID("_SortedPositions");
        static readonly int SortedVelocitiesId = Shader.PropertyToID("_SortedVelocities");
        static readonly int PositionsId = Shader.PropertyToID("_Positions");
        static readonly int VelocitiesId = Shader.PropertyToID("_Velocities");
        static readonly int SortedIndicesId = Shader.PropertyToID("_SortedIndices");
        static readonly int CellOffsetsId = Shader.PropertyToID("_CellOffsets");
        static readonly int CellCountsId = Shader.PropertyToID("_CellCounts");
        static readonly int DensitiesId = Shader.PropertyToID("_Densities");
        static readonly int PressuresId = Shader.PropertyToID("_Pressures");
        static readonly int SpeedsId = Shader.PropertyToID("_Speeds");
        static readonly int AccelerationsId = Shader.PropertyToID("_Accelerations");
        static readonly int BoundaryPositionsId = Shader.PropertyToID("_BoundaryPositions");
        static readonly int BoundaryMassesId = Shader.PropertyToID("_BoundaryMasses");
        static readonly int BoundaryCellOffsetsId = Shader.PropertyToID("_BoundaryCellOffsets");
        static readonly int BoundaryCellCountsId = Shader.PropertyToID("_BoundaryCellCounts");
        static readonly int BoundaryParticleCountId = Shader.PropertyToID("_BoundaryParticleCount");
        static readonly int DomainMinId = Shader.PropertyToID("_DomainMin");
        static readonly int DomainMaxId = Shader.PropertyToID("_DomainMax");
        static readonly int GridResolutionId = Shader.PropertyToID("_GridResolution");
        static readonly int CellSizeId = Shader.PropertyToID("_CellSize");
        static readonly int InverseCellSizeId = Shader.PropertyToID("_InverseCellSize");
        static readonly int CellCountId = Shader.PropertyToID("_CellCount");
        static readonly int ParticleCountId = Shader.PropertyToID("_ParticleCount");
        static readonly int SupportRadiusId = Shader.PropertyToID("_SupportRadius");
        static readonly int KernelNormalizationId = Shader.PropertyToID("_KernelNormalization");
        static readonly int ParticleMassId = Shader.PropertyToID("_ParticleMass");
        static readonly int RestDensityId = Shader.PropertyToID("_RestDensity");
        static readonly int StiffnessId = Shader.PropertyToID("_Stiffness");
        static readonly int KinematicViscosityId = Shader.PropertyToID("_KinematicViscosity");
        static readonly int ParticleRadiusId = Shader.PropertyToID("_ParticleRadius");
        static readonly int DeltaTimeId = Shader.PropertyToID("_DeltaTime");
        static readonly int RestitutionId = Shader.PropertyToID("_Restitution");
        static readonly int GravityId = Shader.PropertyToID("_Gravity");

        readonly ComputeShader shader;
        readonly int boundaryMassKernel;
        readonly int densityKernel;
        readonly int accelerationKernel;
        readonly int integrateKernel;

        readonly ComputeBuffer dummyBoundaryPositions;
        readonly ComputeBuffer dummyBoundaryMasses;
        readonly ComputeBuffer dummyBoundaryCells;

        ParticleSet boundaryParticles;
        NeighborGrid boundaryGrid;
        ComputeBuffer boundaryMasses;
        int boundaryCount;

        public int Capacity { get; }

        public ComputeBuffer Densities { get; private set; }

        public ComputeBuffer Pressures { get; private set; }

        public ComputeBuffer Speeds { get; private set; }

        public ComputeBuffer Accelerations { get; private set; }

        public int BoundaryCount => boundaryCount;

        public ParticleSet BoundaryParticles => boundaryParticles;

        public SesphSolver(ComputeShader shader, int capacity, int dimensions)
        {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));

            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be positive.");
            }

            Capacity = capacity;

            boundaryMassKernel = shader.FindKernel("ComputeBoundaryMasses");
            densityKernel = shader.FindKernel("ComputeDensity");
            accelerationKernel = shader.FindKernel("ComputeAccelerations");
            integrateKernel = shader.FindKernel("Integrate");

            Densities = new ComputeBuffer(capacity, sizeof(float));
            Pressures = new ComputeBuffer(capacity, sizeof(float));
            Speeds = new ComputeBuffer(capacity, sizeof(float));
            Accelerations = new ComputeBuffer(capacity, sizeof(float) * dimensions);

            dummyBoundaryPositions = new ComputeBuffer(1, sizeof(float) * dimensions);
            dummyBoundaryMasses = new ComputeBuffer(1, sizeof(float));
            dummyBoundaryCells = new ComputeBuffer(1, sizeof(uint));
        }

        /// <summary>
        /// Installs a static Akinci boundary. Masses are derived on the GPU from the
        /// boundary's own neighbourhood so a one-layer wall still contributes a full
        /// rest-density half-kernel. The grid must already have been built.
        /// </summary>
        public void SetBoundary(ParticleSet particles, NeighborGrid grid, FluidParameters parameters)
        {
            boundaryParticles = particles;
            boundaryGrid = grid;
            boundaryCount = particles != null ? particles.Count : 0;

            boundaryMasses?.Release();
            boundaryMasses = null;

            if (boundaryCount <= 0)
            {
                return;
            }

            boundaryMasses = new ComputeBuffer(boundaryCount, sizeof(float));
            SetUniforms(parameters, particleCount: 0, deltaTime: 0f);
            BindBoundary(boundaryMassKernel);
            shader.Dispatch(boundaryMassKernel, (boundaryCount + ParticleGroupSize - 1) / ParticleGroupSize, 1, 1);
        }

        public void ComputeDensity(ParticleSet particles, NeighborGrid grid, FluidParameters parameters)
        {
            if (particles.Count == 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count, deltaTime: 0f);
            BindParticleAndGrid(densityKernel, particles, grid);
            BindBoundary(densityKernel);
            shader.SetBuffer(densityKernel, DensitiesId, Densities);
            shader.SetBuffer(densityKernel, PressuresId, Pressures);
            DispatchOver(densityKernel, particles.Count);
        }

        public void ComputeAccelerations(ParticleSet particles, NeighborGrid grid, FluidParameters parameters)
        {
            if (particles.Count == 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count, deltaTime: 0f);
            BindParticleAndGrid(accelerationKernel, particles, grid);
            BindBoundary(accelerationKernel);
            shader.SetBuffer(accelerationKernel, DensitiesId, Densities);
            shader.SetBuffer(accelerationKernel, PressuresId, Pressures);
            shader.SetBuffer(accelerationKernel, AccelerationsId, Accelerations);
            DispatchOver(accelerationKernel, particles.Count);
        }

        public void Integrate(ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            if (particles.Count == 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(integrateKernel, particles, grid);
            shader.SetBuffer(integrateKernel, PositionsId, particles.Positions);
            shader.SetBuffer(integrateKernel, VelocitiesId, particles.Velocities);
            shader.SetBuffer(integrateKernel, SortedIndicesId, grid.SortedIndices);
            shader.SetBuffer(integrateKernel, AccelerationsId, Accelerations);
            shader.SetBuffer(integrateKernel, SpeedsId, Speeds);
            DispatchOver(integrateKernel, particles.Count);
        }

        /// <summary>
        /// One SESPH step on an already-built neighbour grid. The caller is responsible
        /// for rebuilding the grid between steps.
        /// </summary>
        public void Advance(ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            ComputeDensity(particles, grid, parameters);
            ComputeAccelerations(particles, grid, parameters);
            Integrate(particles, grid, parameters, deltaTime);
        }

        public float[] ReadDensities(int count)
        {
            return Read(Densities, count);
        }

        public float[] ReadPressures(int count)
        {
            return Read(Pressures, count);
        }

        public void Dispose()
        {
            Densities?.Release();
            Pressures?.Release();
            Speeds?.Release();
            Accelerations?.Release();

            Densities = null;
            Pressures = null;
            Speeds = null;
            Accelerations = null;

            boundaryMasses?.Release();
            boundaryMasses = null;
            dummyBoundaryPositions?.Release();
            dummyBoundaryMasses?.Release();
            dummyBoundaryCells?.Release();
            boundaryParticles = null;
            boundaryGrid = null;
            boundaryCount = 0;
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

        void BindBoundary(int kernel)
        {
            bool hasBoundary = boundaryCount > 0 && boundaryParticles != null && boundaryGrid != null &&
                               boundaryMasses != null;

            shader.SetInt(BoundaryParticleCountId, hasBoundary ? boundaryCount : 0);
            shader.SetBuffer(kernel, BoundaryPositionsId,
                hasBoundary ? boundaryParticles.SortedPositions : dummyBoundaryPositions);
            shader.SetBuffer(kernel, BoundaryMassesId, hasBoundary ? boundaryMasses : dummyBoundaryMasses);
            shader.SetBuffer(kernel, BoundaryCellOffsetsId,
                hasBoundary ? boundaryGrid.CellOffsets : dummyBoundaryCells);
            shader.SetBuffer(kernel, BoundaryCellCountsId,
                hasBoundary ? boundaryGrid.CellCounts : dummyBoundaryCells);
        }

        void BindParticleAndGrid(int kernel, ParticleSet particles, NeighborGrid grid)
        {
            shader.SetBuffer(kernel, SortedPositionsId, particles.SortedPositions);
            shader.SetBuffer(kernel, SortedVelocitiesId, particles.SortedVelocities);
            shader.SetBuffer(kernel, CellOffsetsId, grid.CellOffsets);
            shader.SetBuffer(kernel, CellCountsId, grid.CellCounts);
        }

        void SetUniforms(FluidParameters parameters, int particleCount, float deltaTime)
        {
            Bounds domain = parameters.Domain;
            int3 resolution = parameters.GridResolution;
            float3 gravity = parameters.Gravity;

            shader.SetVector(DomainMinId, new Vector4(domain.min.x, domain.min.y, domain.min.z, 0f));
            shader.SetVector(DomainMaxId, new Vector4(domain.max.x, domain.max.y, domain.max.z, 0f));
            shader.SetVector(GridResolutionId, new Vector4(resolution.x, resolution.y, resolution.z, 0f));
            shader.SetFloat(CellSizeId, parameters.CellSize);
            shader.SetFloat(InverseCellSizeId, 1f / parameters.CellSize);
            shader.SetInt(CellCountId, parameters.CellCount);
            shader.SetInt(ParticleCountId, particleCount);
            shader.SetFloat(SupportRadiusId, parameters.SupportRadius);
            shader.SetFloat(KernelNormalizationId, parameters.KernelNormalization);
            shader.SetFloat(ParticleMassId, parameters.ParticleMass);
            shader.SetFloat(RestDensityId, parameters.RestDensity);
            shader.SetFloat(StiffnessId, parameters.Stiffness);
            shader.SetFloat(KinematicViscosityId, parameters.KinematicViscosity);
            shader.SetFloat(ParticleRadiusId, parameters.ParticleRadius);
            shader.SetFloat(DeltaTimeId, deltaTime);
            shader.SetFloat(RestitutionId, 0f);
            shader.SetVector(GravityId, new Vector4(gravity.x, gravity.y, gravity.z, 0f));
        }

        void DispatchOver(int kernel, int elementCount)
        {
            shader.Dispatch(kernel, (elementCount + ParticleGroupSize - 1) / ParticleGroupSize, 1, 1);
        }
    }
}
