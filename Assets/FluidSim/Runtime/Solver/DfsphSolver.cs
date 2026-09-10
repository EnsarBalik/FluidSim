using System;
using FluidSim.Core;
using FluidSim.Neighbors;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Solver
{
    /// <summary>
    /// Divergence-free SPH. Matches Algorithms 4–6 of the 2019 tutorial: a constant-density
    /// Jacobi corrector before advection, then a divergence-free corrector after the
    /// neighbourhood is rebuilt at the new positions.
    ///
    /// The state-equation solver stays in the project so the inspector can roll back to it.
    /// Walls are the same Akinci one-layer scheme; the kinematic box clamp is still the leak net.
    /// </summary>
    public sealed class DfsphSolver : IDisposable
    {
        public const int DefaultDensityIterations = 3;
        public const int DefaultDivergenceIterations = 2;

        const int ParticleGroupSize = 64;

        static readonly int SortedPositionsId = Shader.PropertyToID("_SortedPositions");
        static readonly int SortedVelocitiesId = Shader.PropertyToID("_SortedVelocities");
        static readonly int PositionsId = Shader.PropertyToID("_Positions");
        static readonly int VelocitiesId = Shader.PropertyToID("_Velocities");
        static readonly int PredictedVelocitiesId = Shader.PropertyToID("_PredictedVelocities");
        static readonly int ViscosityRhsId = Shader.PropertyToID("_ViscosityRhs");
        static readonly int ViscosityVelocitiesId = Shader.PropertyToID("_ViscosityVelocities");
        static readonly int ImplicitViscosityIterationsId = Shader.PropertyToID("_ImplicitViscosityIterations");
        static readonly int SortedIndicesId = Shader.PropertyToID("_SortedIndices");
        static readonly int CellOffsetsId = Shader.PropertyToID("_CellOffsets");
        static readonly int CellCountsId = Shader.PropertyToID("_CellCounts");
        static readonly int DensitiesId = Shader.PropertyToID("_Densities");
        static readonly int PredictedDensitiesId = Shader.PropertyToID("_PredictedDensities");
        static readonly int PressuresId = Shader.PropertyToID("_Pressures");
        static readonly int StiffnessFactorsId = Shader.PropertyToID("_StiffnessFactors");
        static readonly int SpeedsId = Shader.PropertyToID("_Speeds");
        static readonly int BoundaryPositionsId = Shader.PropertyToID("_BoundaryPositions");
        static readonly int BoundaryMassesId = Shader.PropertyToID("_BoundaryMasses");
        static readonly int BoundaryCellOffsetsId = Shader.PropertyToID("_BoundaryCellOffsets");
        static readonly int BoundaryCellCountsId = Shader.PropertyToID("_BoundaryCellCounts");
        static readonly int BoundaryParticleCountId = Shader.PropertyToID("_BoundaryParticleCount");
        static readonly int RigidPositionsId = Shader.PropertyToID("_RigidPositions");
        static readonly int RigidVelocitiesId = Shader.PropertyToID("_RigidVelocities");
        static readonly int RigidMassesId = Shader.PropertyToID("_RigidMasses");
        static readonly int RigidMassesOutId = Shader.PropertyToID("_RigidMassesOut");
        static readonly int RigidCellOffsetsId = Shader.PropertyToID("_RigidCellOffsets");
        static readonly int RigidCellCountsId = Shader.PropertyToID("_RigidCellCounts");
        static readonly int RigidForcesId = Shader.PropertyToID("_RigidForces");
        static readonly int UnsortedRigidMassesId = Shader.PropertyToID("_UnsortedRigidMasses");
        static readonly int RigidSortedIndicesId = Shader.PropertyToID("_RigidSortedIndices");
        static readonly int RigidReducedId = Shader.PropertyToID("_RigidReduced");
        static readonly int RigidCenterOfMassId = Shader.PropertyToID("_RigidCenterOfMass");
        static readonly int RigidParticleCountId = Shader.PropertyToID("_RigidParticleCount");
        static readonly int RigidAccumulateModeId = Shader.PropertyToID("_RigidAccumulateMode");
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
        static readonly int KinematicViscosityId = Shader.PropertyToID("_KinematicViscosity");
        static readonly int SurfaceTensionId = Shader.PropertyToID("_SurfaceTension");
        static readonly int AdhesionId = Shader.PropertyToID("_Adhesion");
        static readonly int SurfaceNormalsId = Shader.PropertyToID("_SurfaceNormals");
        static readonly int AngularVelocitiesId = Shader.PropertyToID("_AngularVelocities");
        static readonly int SortedAngularVelocitiesId = Shader.PropertyToID("_SortedAngularVelocities");
        static readonly int VorticityId = Shader.PropertyToID("_Vorticity");
        static readonly int ViscosityOmegaId = Shader.PropertyToID("_ViscosityOmega");
        static readonly int InertiaInverseId = Shader.PropertyToID("_InertiaInverse");
        static readonly int ParticleRadiusId = Shader.PropertyToID("_ParticleRadius");
        static readonly int DeltaTimeId = Shader.PropertyToID("_DeltaTime");
        static readonly int RestitutionId = Shader.PropertyToID("_Restitution");
        static readonly int GravityId = Shader.PropertyToID("_Gravity");

        readonly ComputeShader shader;
        readonly int boundaryMassKernel;
        readonly int densityKernel;
        readonly int factorKernel;
        readonly int surfaceNormalsKernel;
        readonly int reorderAngularKernel;
        readonly int predictVelocityKernel;
        readonly int predictDensityKernel;
        readonly int divergenceKernel;
        readonly int applyPressureKernel;
        readonly int integrateKernel;
        readonly int seedPredictedKernel;
        readonly int finalizeKernel;
        readonly int accumulateRigidKernel;
        readonly int seedViscosityKernel;
        readonly int jacobiViscosityKernel;
        readonly int commitViscosityKernel;
        readonly int scatterRigidMassesKernel;
        readonly int reorderRigidMassesKernel;
        readonly int clearRigidForcesKernel;
        readonly int reduceRigidForcesKernel;

        readonly ComputeBuffer dummyBoundaryPositions;
        readonly ComputeBuffer dummyBoundaryMasses;
        readonly ComputeBuffer dummyBoundaryCells;
        readonly ComputeBuffer dummyRigidPositions;
        readonly ComputeBuffer dummyRigidVelocities;
        readonly ComputeBuffer dummyRigidMasses;
        readonly ComputeBuffer dummyRigidCells;
        readonly ComputeBuffer dummyRigidForces;

        ParticleSet boundaryParticles;
        NeighborGrid boundaryGrid;
        ComputeBuffer boundaryMasses;
        int boundaryCount;

        ParticleSet rigidParticles;
        NeighborGrid rigidGrid;
        ComputeBuffer rigidMasses;
        ComputeBuffer rigidUnsortedMasses;
        ComputeBuffer rigidForces;
        ComputeBuffer rigidReduced;
        int rigidCount;

        public int Capacity { get; }

        public ComputeBuffer Densities { get; private set; }

        public ComputeBuffer PredictedDensities { get; private set; }

        public ComputeBuffer Pressures { get; private set; }

        public ComputeBuffer StiffnessFactors { get; private set; }

        public ComputeBuffer Speeds { get; private set; }

        public ComputeBuffer PredictedVelocities { get; private set; }

        public ComputeBuffer ViscosityRhs { get; private set; }

        public ComputeBuffer ViscosityVelocities { get; private set; }

        public ComputeBuffer SurfaceNormals { get; private set; }

        public ComputeBuffer AngularVelocities { get; private set; }

        public ComputeBuffer SortedAngularVelocities { get; private set; }

        public int BoundaryCount => boundaryCount;

        public ParticleSet BoundaryParticles => boundaryParticles;

        public int RigidCount => rigidCount;

        public DfsphSolver(ComputeShader shader, int capacity, int dimensions)
        {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));

            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be positive.");
            }

            Capacity = capacity;

            boundaryMassKernel = shader.FindKernel("ComputeBoundaryMasses");
            densityKernel = shader.FindKernel("ComputeDensity");
            factorKernel = shader.FindKernel("ComputeFactor");
            surfaceNormalsKernel = shader.FindKernel("ComputeSurfaceNormals");
            reorderAngularKernel = shader.FindKernel("ReorderAngularVelocities");
            predictVelocityKernel = shader.FindKernel("PredictVelocity");
            predictDensityKernel = shader.FindKernel("PredictDensity");
            divergenceKernel = shader.FindKernel("ComputeDivergence");
            applyPressureKernel = shader.FindKernel("ApplyPressure");
            integrateKernel = shader.FindKernel("IntegratePositions");
            seedPredictedKernel = shader.FindKernel("SeedPredictedVelocities");
            finalizeKernel = shader.FindKernel("FinalizeVelocities");
            accumulateRigidKernel = shader.FindKernel("AccumulateRigidForces");
            seedViscosityKernel = shader.FindKernel("SeedViscosityState");
            jacobiViscosityKernel = shader.FindKernel("JacobiViscosity");
            commitViscosityKernel = shader.FindKernel("CommitViscosityIterate");
            scatterRigidMassesKernel = shader.FindKernel("ScatterRigidMasses");
            reorderRigidMassesKernel = shader.FindKernel("ReorderRigidMasses");
            clearRigidForcesKernel = shader.FindKernel("ClearRigidForces");
            reduceRigidForcesKernel = shader.FindKernel("ReduceRigidForces");

            Densities = new ComputeBuffer(capacity, sizeof(float));
            PredictedDensities = new ComputeBuffer(capacity, sizeof(float));
            Pressures = new ComputeBuffer(capacity, sizeof(float));
            StiffnessFactors = new ComputeBuffer(capacity, sizeof(float));
            Speeds = new ComputeBuffer(capacity, sizeof(float));
            PredictedVelocities = new ComputeBuffer(capacity, sizeof(float) * dimensions);
            ViscosityRhs = new ComputeBuffer(capacity, sizeof(float) * dimensions);
            ViscosityVelocities = new ComputeBuffer(capacity, sizeof(float) * dimensions);
            SurfaceNormals = new ComputeBuffer(capacity, sizeof(float) * dimensions);
            AngularVelocities = new ComputeBuffer(capacity, sizeof(float) * 3);
            SortedAngularVelocities = new ComputeBuffer(capacity, sizeof(float) * 3);

            Densities.SetData(new float[capacity]);
            PredictedDensities.SetData(new float[capacity]);
            Pressures.SetData(new float[capacity]);
            StiffnessFactors.SetData(new float[capacity]);
            Speeds.SetData(new float[capacity]);
            SurfaceNormals.SetData(new float[capacity * dimensions]);
            AngularVelocities.SetData(new float[capacity * 3]);
            SortedAngularVelocities.SetData(new float[capacity * 3]);

            dummyBoundaryPositions = new ComputeBuffer(1, sizeof(float) * dimensions);
            dummyBoundaryMasses = new ComputeBuffer(1, sizeof(float));
            dummyBoundaryCells = new ComputeBuffer(1, sizeof(uint));
            dummyRigidPositions = new ComputeBuffer(1, sizeof(float) * dimensions);
            dummyRigidVelocities = new ComputeBuffer(1, sizeof(float) * dimensions);
            dummyRigidMasses = new ComputeBuffer(1, sizeof(float));
            dummyRigidCells = new ComputeBuffer(1, sizeof(uint));
            dummyRigidForces = new ComputeBuffer(1, sizeof(float) * 3);
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

        /// <summary>
        /// Installs a dynamic Akinci rigid surface. Particle masses are captured once
        /// from the rigid's own neighbourhood, then only reordered after a grid rebuild.
        /// </summary>
        public void SetRigid(SphRigidBody body, FluidParameters parameters)
        {
            if (body == null || body.Count <= 0)
            {
                rigidParticles = null;
                rigidGrid = null;
                rigidMasses = null;
                rigidUnsortedMasses = null;
                rigidForces = null;
                rigidReduced = null;
                rigidCount = 0;
                return;
            }

            rigidParticles = body.Particles;
            rigidGrid = body.Grid;
            rigidMasses = body.Masses;
            rigidUnsortedMasses = body.UnsortedMasses;
            rigidForces = body.Forces;
            rigidReduced = body.ReducedForceAndTorque;
            rigidCount = body.Count;

            SetUniforms(parameters, particleCount: 0, deltaTime: 0f);
            if (!body.MassesCaptured)
            {
                shader.SetInt(BoundaryParticleCountId, rigidCount);
                shader.SetBuffer(boundaryMassKernel, BoundaryPositionsId, rigidParticles.SortedPositions);
                shader.SetBuffer(boundaryMassKernel, BoundaryMassesId, rigidMasses);
                shader.SetBuffer(boundaryMassKernel, BoundaryCellOffsetsId, rigidGrid.CellOffsets);
                shader.SetBuffer(boundaryMassKernel, BoundaryCellCountsId, rigidGrid.CellCounts);
                shader.Dispatch(boundaryMassKernel, (rigidCount + ParticleGroupSize - 1) / ParticleGroupSize, 1, 1);
                ScatterRigidMasses();
                body.MassesCaptured = true;
                return;
            }

            ReorderRigidMasses();
        }

        public void ClearRigidForces()
        {
            if (rigidCount <= 0 || rigidForces == null)
            {
                return;
            }

            shader.SetInt(RigidParticleCountId, rigidCount);
            shader.SetBuffer(clearRigidForcesKernel, RigidForcesId, rigidForces);
            DispatchOver(clearRigidForcesKernel, rigidCount);
        }

        public float3[] ReadRigidForces()
        {
            var forces = new float3[rigidCount];
            if (rigidCount > 0 && rigidForces != null)
            {
                rigidForces.GetData(forces, 0, 0, rigidCount);
            }

            return forces;
        }

        public void ReadRigidForceAndTorque(float3 centerOfMass, out float3 force, out float3 torque)
        {
            force = float3.zero;
            torque = float3.zero;
            if (rigidCount <= 0 || rigidForces == null || rigidParticles == null || rigidReduced == null)
            {
                return;
            }

            shader.SetInt(RigidParticleCountId, rigidCount);
            shader.SetVector(RigidCenterOfMassId, new Vector4(centerOfMass.x, centerOfMass.y, centerOfMass.z, 0f));
            shader.SetBuffer(reduceRigidForcesKernel, RigidPositionsId, rigidParticles.SortedPositions);
            shader.SetBuffer(reduceRigidForcesKernel, RigidForcesId, rigidForces);
            shader.SetBuffer(reduceRigidForcesKernel, RigidReducedId, rigidReduced);
            shader.Dispatch(reduceRigidForcesKernel, 1, 1, 1);

            var reduced = new float3[2];
            rigidReduced.GetData(reduced, 0, 0, 2);
            force = reduced[0];
            torque = reduced[1];
        }

        void ScatterRigidMasses()
        {
            shader.SetInt(RigidParticleCountId, rigidCount);
            shader.SetBuffer(scatterRigidMassesKernel, RigidMassesId, rigidMasses);
            shader.SetBuffer(scatterRigidMassesKernel, UnsortedRigidMassesId, rigidUnsortedMasses);
            shader.SetBuffer(scatterRigidMassesKernel, RigidSortedIndicesId, rigidGrid.SortedIndices);
            DispatchOver(scatterRigidMassesKernel, rigidCount);
        }

        void ReorderRigidMasses()
        {
            shader.SetInt(RigidParticleCountId, rigidCount);
            shader.SetBuffer(reorderRigidMassesKernel, RigidMassesOutId, rigidMasses);
            shader.SetBuffer(reorderRigidMassesKernel, UnsortedRigidMassesId, rigidUnsortedMasses);
            shader.SetBuffer(reorderRigidMassesKernel, RigidSortedIndicesId, rigidGrid.SortedIndices);
            DispatchOver(reorderRigidMassesKernel, rigidCount);
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
            BindRigid(densityKernel);
            shader.SetBuffer(densityKernel, DensitiesId, Densities);
            DispatchOver(densityKernel, particles.Count);
        }

        /// <summary>
        /// Density, stiffness factor, non-pressure predict, constant-density iterations,
        /// then <c>x += Δt v*</c>. The caller must rebuild the fluid grid before
        /// <see cref="EndStep"/>.
        /// </summary>
        public void BeginStep(
            ParticleSet particles,
            NeighborGrid grid,
            FluidParameters parameters,
            float deltaTime,
            int densityIterations = DefaultDensityIterations)
        {
            if (particles.Count == 0)
            {
                return;
            }

            ComputeDensityAndFactor(particles, grid, parameters, deltaTime);
            if (parameters.SurfaceTension > 0f)
            {
                ComputeSurfaceNormals(particles, grid, parameters);
            }

            if (parameters.Vorticity > 0f || parameters.ViscosityOmega > 0f)
            {
                ReorderAngularVelocities(particles, grid, parameters);
            }

            PredictNonPressure(particles, grid, parameters, deltaTime);
            SolveImplicitViscosity(particles, grid, parameters, deltaTime);
            AccumulateRigidForces(particles, grid, parameters, deltaTime, pressure: false);

            int iterations = math.max(densityIterations, 1);
            for (int i = 0; i < iterations; i++)
            {
                DispatchPredictDensity(particles, grid, parameters, deltaTime);
                DispatchApplyPressure(particles, grid, parameters, deltaTime);
                AccumulateRigidForces(particles, grid, parameters, deltaTime, pressure: true);
            }

            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(integrateKernel, particles, grid);
            shader.SetBuffer(integrateKernel, PredictedVelocitiesId, PredictedVelocities);
            shader.SetBuffer(integrateKernel, PositionsId, particles.Positions);
            shader.SetBuffer(integrateKernel, VelocitiesId, particles.Velocities);
            shader.SetBuffer(integrateKernel, SortedIndicesId, grid.SortedIndices);
            DispatchOver(integrateKernel, particles.Count);
        }

        /// <summary>
        /// Recompute density and k at the new positions, then project the predicted
        /// velocities onto a divergence-free field and commit them.
        /// </summary>
        public void EndStep(
            ParticleSet particles,
            NeighborGrid grid,
            FluidParameters parameters,
            float deltaTime,
            int divergenceIterations = DefaultDivergenceIterations)
        {
            if (particles.Count == 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(seedPredictedKernel, particles, grid);
            shader.SetBuffer(seedPredictedKernel, PredictedVelocitiesId, PredictedVelocities);
            DispatchOver(seedPredictedKernel, particles.Count);

            ComputeDensityAndFactor(particles, grid, parameters, deltaTime);

            int iterations = math.max(divergenceIterations, 1);
            for (int i = 0; i < iterations; i++)
            {
                DispatchDivergence(particles, grid, parameters, deltaTime);
                DispatchApplyPressure(particles, grid, parameters, deltaTime);
                AccumulateRigidForces(particles, grid, parameters, deltaTime, pressure: true);
            }

            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(finalizeKernel, particles, grid);
            shader.SetBuffer(finalizeKernel, PredictedVelocitiesId, PredictedVelocities);
            shader.SetBuffer(finalizeKernel, VelocitiesId, particles.Velocities);
            shader.SetBuffer(finalizeKernel, SortedIndicesId, grid.SortedIndices);
            shader.SetBuffer(finalizeKernel, SpeedsId, Speeds);
            DispatchOver(finalizeKernel, particles.Count);
        }

        /// <summary>
        /// One full DFSPH step. <paramref name="rebuildNeighborhood"/> must rebuild the
        /// fluid grid from the positions written by <see cref="BeginStep"/>.
        /// </summary>
        public void Advance(
            ParticleSet particles,
            NeighborGrid grid,
            FluidParameters parameters,
            float deltaTime,
            Action rebuildNeighborhood,
            int densityIterations = DefaultDensityIterations,
            int divergenceIterations = DefaultDivergenceIterations)
        {
            if (rebuildNeighborhood == null)
            {
                throw new ArgumentNullException(nameof(rebuildNeighborhood));
            }

            BeginStep(particles, grid, parameters, deltaTime, densityIterations);
            rebuildNeighborhood();
            EndStep(particles, grid, parameters, deltaTime, divergenceIterations);
        }

        public float[] ReadDensities(int count)
        {
            return Read(Densities, count);
        }

        public float[] ReadPressures(int count)
        {
            return Read(Pressures, count);
        }

        public float[] ReadStiffnessFactors(int count)
        {
            return Read(StiffnessFactors, count);
        }

        public float[] ReadSurfaceNormalLengths(int count)
        {
            var lengths = new float[count];
            if (count <= 0)
            {
                return lengths;
            }

            if (SurfaceNormals.stride == sizeof(float) * 3)
            {
                var vectors = new float3[count];
                SurfaceNormals.GetData(vectors, 0, 0, count);
                for (int i = 0; i < count; i++)
                {
                    lengths[i] = math.length(vectors[i]);
                }
            }
            else
            {
                var vectors = new float2[count];
                SurfaceNormals.GetData(vectors, 0, 0, count);
                for (int i = 0; i < count; i++)
                {
                    lengths[i] = math.length(vectors[i]);
                }
            }

            return lengths;
        }

        public float[] ReadAngularSpeeds(int count)
        {
            var lengths = new float[count];
            if (count <= 0)
            {
                return lengths;
            }

            var vectors = new float3[count];
            AngularVelocities.GetData(vectors, 0, 0, count);
            for (int i = 0; i < count; i++)
            {
                lengths[i] = math.length(vectors[i]);
            }

            return lengths;
        }

        public void Dispose()
        {
            Densities?.Release();
            PredictedDensities?.Release();
            Pressures?.Release();
            StiffnessFactors?.Release();
            Speeds?.Release();
            PredictedVelocities?.Release();
            ViscosityRhs?.Release();
            ViscosityVelocities?.Release();
            SurfaceNormals?.Release();
            AngularVelocities?.Release();
            SortedAngularVelocities?.Release();

            Densities = null;
            PredictedDensities = null;
            Pressures = null;
            StiffnessFactors = null;
            Speeds = null;
            PredictedVelocities = null;
            ViscosityRhs = null;
            ViscosityVelocities = null;
            SurfaceNormals = null;
            AngularVelocities = null;
            SortedAngularVelocities = null;

            boundaryMasses?.Release();
            boundaryMasses = null;
            dummyBoundaryPositions?.Release();
            dummyBoundaryMasses?.Release();
            dummyBoundaryCells?.Release();
            dummyRigidPositions?.Release();
            dummyRigidVelocities?.Release();
            dummyRigidMasses?.Release();
            dummyRigidCells?.Release();
            dummyRigidForces?.Release();
            boundaryParticles = null;
            boundaryGrid = null;
            boundaryCount = 0;
            rigidParticles = null;
            rigidGrid = null;
            rigidMasses = null;
            rigidUnsortedMasses = null;
            rigidForces = null;
            rigidReduced = null;
            rigidCount = 0;
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

        void ComputeDensityAndFactor(
            ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            ComputeDensity(particles, grid, parameters);

            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(factorKernel, particles, grid);
            BindBoundary(factorKernel);
            BindRigid(factorKernel);
            shader.SetBuffer(factorKernel, DensitiesId, Densities);
            shader.SetBuffer(factorKernel, StiffnessFactorsId, StiffnessFactors);
            DispatchOver(factorKernel, particles.Count);
        }

        public void ComputeSurfaceNormals(ParticleSet particles, NeighborGrid grid, FluidParameters parameters)
        {
            SetUniforms(parameters, particles.Count, deltaTime: 0f);
            BindParticleAndGrid(surfaceNormalsKernel, particles, grid);
            shader.SetBuffer(surfaceNormalsKernel, DensitiesId, Densities);
            shader.SetBuffer(surfaceNormalsKernel, SurfaceNormalsId, SurfaceNormals);
            DispatchOver(surfaceNormalsKernel, particles.Count);
        }

        public void ReorderAngularVelocities(ParticleSet particles, NeighborGrid grid, FluidParameters parameters)
        {
            SetUniforms(parameters, particles.Count, deltaTime: 0f);
            shader.SetBuffer(reorderAngularKernel, AngularVelocitiesId, AngularVelocities);
            shader.SetBuffer(reorderAngularKernel, SortedAngularVelocitiesId, SortedAngularVelocities);
            shader.SetBuffer(reorderAngularKernel, SortedIndicesId, grid.SortedIndices);
            DispatchOver(reorderAngularKernel, particles.Count);
        }

        void PredictNonPressure(
            ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(predictVelocityKernel, particles, grid);
            BindBoundary(predictVelocityKernel);
            BindRigid(predictVelocityKernel);
            shader.SetBuffer(predictVelocityKernel, DensitiesId, Densities);
            shader.SetBuffer(predictVelocityKernel, SurfaceNormalsId, SurfaceNormals);
            shader.SetBuffer(predictVelocityKernel, PredictedVelocitiesId, PredictedVelocities);
            shader.SetBuffer(predictVelocityKernel, SortedAngularVelocitiesId, SortedAngularVelocities);
            shader.SetBuffer(predictVelocityKernel, AngularVelocitiesId, AngularVelocities);
            shader.SetBuffer(predictVelocityKernel, SortedIndicesId, grid.SortedIndices);
            DispatchOver(predictVelocityKernel, particles.Count);
        }

        void DispatchPredictDensity(
            ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(predictDensityKernel, particles, grid);
            BindBoundary(predictDensityKernel);
            BindRigid(predictDensityKernel);
            shader.SetBuffer(predictDensityKernel, DensitiesId, Densities);
            shader.SetBuffer(predictDensityKernel, PredictedDensitiesId, PredictedDensities);
            shader.SetBuffer(predictDensityKernel, PredictedVelocitiesId, PredictedVelocities);
            shader.SetBuffer(predictDensityKernel, StiffnessFactorsId, StiffnessFactors);
            shader.SetBuffer(predictDensityKernel, PressuresId, Pressures);
            DispatchOver(predictDensityKernel, particles.Count);
        }

        void DispatchDivergence(
            ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(divergenceKernel, particles, grid);
            BindBoundary(divergenceKernel);
            BindRigid(divergenceKernel);
            shader.SetBuffer(divergenceKernel, PredictedVelocitiesId, PredictedVelocities);
            shader.SetBuffer(divergenceKernel, StiffnessFactorsId, StiffnessFactors);
            shader.SetBuffer(divergenceKernel, PressuresId, Pressures);
            DispatchOver(divergenceKernel, particles.Count);
        }

        void DispatchApplyPressure(
            ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(applyPressureKernel, particles, grid);
            BindBoundary(applyPressureKernel);
            BindRigid(applyPressureKernel);
            shader.SetBuffer(applyPressureKernel, DensitiesId, Densities);
            shader.SetBuffer(applyPressureKernel, PressuresId, Pressures);
            shader.SetBuffer(applyPressureKernel, PredictedVelocitiesId, PredictedVelocities);
            DispatchOver(applyPressureKernel, particles.Count);
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

        void BindRigid(int kernel)
        {
            bool hasRigid = rigidCount > 0 && rigidParticles != null && rigidGrid != null &&
                            rigidMasses != null && rigidForces != null;

            shader.SetInt(RigidParticleCountId, hasRigid ? rigidCount : 0);
            shader.SetBuffer(kernel, RigidPositionsId,
                hasRigid ? rigidParticles.SortedPositions : dummyRigidPositions);
            shader.SetBuffer(kernel, RigidVelocitiesId,
                hasRigid ? rigidParticles.SortedVelocities : dummyRigidVelocities);
            shader.SetBuffer(kernel, RigidMassesId, hasRigid ? rigidMasses : dummyRigidMasses);
            shader.SetBuffer(kernel, RigidCellOffsetsId,
                hasRigid ? rigidGrid.CellOffsets : dummyRigidCells);
            shader.SetBuffer(kernel, RigidCellCountsId,
                hasRigid ? rigidGrid.CellCounts : dummyRigidCells);
            shader.SetBuffer(kernel, RigidForcesId, hasRigid ? rigidForces : dummyRigidForces);
        }

        void AccumulateRigidForces(
            ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime, bool pressure)
        {
            if (rigidCount <= 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count, deltaTime);
            BindParticleAndGrid(accumulateRigidKernel, particles, grid);
            BindRigid(accumulateRigidKernel);
            shader.SetInt(RigidAccumulateModeId, pressure ? 1 : 0);
            shader.SetBuffer(accumulateRigidKernel, DensitiesId, Densities);
            shader.SetBuffer(accumulateRigidKernel, PressuresId, Pressures);
            shader.SetBuffer(accumulateRigidKernel, PredictedVelocitiesId, PredictedVelocities);
            DispatchOver(accumulateRigidKernel, rigidCount);
        }

        void SolveImplicitViscosity(
            ParticleSet particles, NeighborGrid grid, FluidParameters parameters, float deltaTime)
        {
            int iterations = parameters.ImplicitViscosityIterations;
            if (iterations <= 0)
            {
                return;
            }

            SetUniforms(parameters, particles.Count, deltaTime);
            shader.SetBuffer(seedViscosityKernel, PredictedVelocitiesId, PredictedVelocities);
            shader.SetBuffer(seedViscosityKernel, ViscosityRhsId, ViscosityRhs);
            shader.SetBuffer(seedViscosityKernel, ViscosityVelocitiesId, ViscosityVelocities);
            DispatchOver(seedViscosityKernel, particles.Count);

            for (int i = 0; i < iterations; i++)
            {
                BindParticleAndGrid(jacobiViscosityKernel, particles, grid);
                BindBoundary(jacobiViscosityKernel);
                BindRigid(jacobiViscosityKernel);
                shader.SetBuffer(jacobiViscosityKernel, DensitiesId, Densities);
                shader.SetBuffer(jacobiViscosityKernel, PredictedVelocitiesId, PredictedVelocities);
                shader.SetBuffer(jacobiViscosityKernel, ViscosityRhsId, ViscosityRhs);
                shader.SetBuffer(jacobiViscosityKernel, ViscosityVelocitiesId, ViscosityVelocities);
                DispatchOver(jacobiViscosityKernel, particles.Count);

                if (i + 1 >= iterations)
                {
                    break;
                }

                shader.SetBuffer(commitViscosityKernel, PredictedVelocitiesId, PredictedVelocities);
                shader.SetBuffer(commitViscosityKernel, ViscosityVelocitiesId, ViscosityVelocities);
                DispatchOver(commitViscosityKernel, particles.Count);
            }
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
            shader.SetFloat(KinematicViscosityId, parameters.KinematicViscosity);
            shader.SetFloat(ImplicitViscosityIterationsId, parameters.ImplicitViscosityIterations);
            shader.SetFloat(SurfaceTensionId, parameters.SurfaceTension);
            shader.SetFloat(AdhesionId, parameters.Adhesion);
            shader.SetFloat(VorticityId, parameters.Vorticity);
            shader.SetFloat(ViscosityOmegaId, parameters.ViscosityOmega);
            shader.SetFloat(InertiaInverseId, parameters.InertiaInverse);
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
