using System;
using FluidSim.Core;
using FluidSim.Neighbors;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Rendering
{
    /// <summary>
    /// GPU post-process for Ihmsen et al. 2012 spray / foam / bubbles.
    /// Generation uses the fluid neighbourhood already built for the solver;
    /// diffuse particles have no inter-particle forces.
    /// </summary>
    public sealed class DiffuseMaterialSystem : IDisposable
    {
        const int ParticleGroupSize = 64;

        static readonly int SortedPositionsId = Shader.PropertyToID("_SortedPositions");
        static readonly int SortedVelocitiesId = Shader.PropertyToID("_SortedVelocities");
        static readonly int CellOffsetsId = Shader.PropertyToID("_CellOffsets");
        static readonly int CellCountsId = Shader.PropertyToID("_CellCounts");
        static readonly int SpawnWeightId = Shader.PropertyToID("_SpawnWeight");
        static readonly int DiffusePositionsId = Shader.PropertyToID("_DiffusePositions");
        static readonly int DiffuseVelocitiesId = Shader.PropertyToID("_DiffuseVelocities");
        static readonly int DiffuseLifeId = Shader.PropertyToID("_DiffuseLife");
        static readonly int DiffuseKindId = Shader.PropertyToID("_DiffuseKind");
        static readonly int DiffuseOccupiedId = Shader.PropertyToID("_DiffuseOccupied");
        static readonly int DomainMinId = Shader.PropertyToID("_DomainMin");
        static readonly int DomainMinBoundsId = Shader.PropertyToID("_DomainMinBounds");
        static readonly int DomainMaxBoundsId = Shader.PropertyToID("_DomainMaxBounds");
        static readonly int GridResolutionId = Shader.PropertyToID("_GridResolution");
        static readonly int CellSizeId = Shader.PropertyToID("_CellSize");
        static readonly int InverseCellSizeId = Shader.PropertyToID("_InverseCellSize");
        static readonly int CellCountId = Shader.PropertyToID("_CellCount");
        static readonly int ParticleCountId = Shader.PropertyToID("_ParticleCount");
        static readonly int SupportRadiusId = Shader.PropertyToID("_SupportRadius");
        static readonly int KernelNormalizationId = Shader.PropertyToID("_KernelNormalization");
        static readonly int ParticleRadiusId = Shader.PropertyToID("_ParticleRadius");
        static readonly int ParticleMassId = Shader.PropertyToID("_ParticleMass");
        static readonly int DeltaTimeId = Shader.PropertyToID("_DeltaTime");
        static readonly int FrameId = Shader.PropertyToID("_Frame");
        static readonly int MaxDiffuseId = Shader.PropertyToID("_MaxDiffuse");
        static readonly int PlaneDepthId = Shader.PropertyToID("_PlaneDepth");
        static readonly int GravityId = Shader.PropertyToID("_Gravity");
        static readonly int TrappedAirMinId = Shader.PropertyToID("_TrappedAirMin");
        static readonly int TrappedAirMaxId = Shader.PropertyToID("_TrappedAirMax");
        static readonly int WaveCrestMinId = Shader.PropertyToID("_WaveCrestMin");
        static readonly int WaveCrestMaxId = Shader.PropertyToID("_WaveCrestMax");
        static readonly int EnergyMinId = Shader.PropertyToID("_EnergyMin");
        static readonly int EnergyMaxId = Shader.PropertyToID("_EnergyMax");
        static readonly int TrappedAirRateId = Shader.PropertyToID("_TrappedAirRate");
        static readonly int WaveCrestRateId = Shader.PropertyToID("_WaveCrestRate");
        static readonly int FoamLifeMinId = Shader.PropertyToID("_FoamLifeMin");
        static readonly int FoamLifeMaxId = Shader.PropertyToID("_FoamLifeMax");
        static readonly int BubbleBuoyancyId = Shader.PropertyToID("_BubbleBuoyancy");
        static readonly int BubbleDragId = Shader.PropertyToID("_BubbleDrag");

        readonly ComputeShader shader;
        readonly int evaluateKernel;
        readonly int spawnKernel;
        readonly int advectKernel;
        uint frame;

        public int Capacity { get; }

        public int FluidCount { get; }

        public ComputeBuffer Positions { get; private set; }

        public ComputeBuffer Velocities { get; private set; }

        public ComputeBuffer Life { get; private set; }

        public ComputeBuffer Kind { get; private set; }

        public ComputeBuffer Occupied { get; private set; }

        public ComputeBuffer SpawnWeight { get; private set; }

        public DiffuseMaterialSystem(ComputeShader shader, int fluidCapacity, int maxDiffuse)
        {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));
            if (fluidCapacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(fluidCapacity));
            }

            FluidCount = fluidCapacity;
            Capacity = math.max(maxDiffuse, 1);
            evaluateKernel = shader.FindKernel("EvaluatePotential");
            spawnKernel = shader.FindKernel("SpawnDiffuse");
            advectKernel = shader.FindKernel("AdvectDiffuse");

            SpawnWeight = new ComputeBuffer(fluidCapacity, sizeof(float));
            Positions = new ComputeBuffer(Capacity, sizeof(float) * 3);
            Velocities = new ComputeBuffer(Capacity, sizeof(float) * 3);
            Life = new ComputeBuffer(Capacity, sizeof(float));
            Kind = new ComputeBuffer(Capacity, sizeof(uint));
            Occupied = new ComputeBuffer(Capacity, sizeof(uint));
            Occupied.SetData(new uint[Capacity]);
            Life.SetData(new float[Capacity]);
        }

        public void Step(
            ParticleSet particles,
            NeighborGrid grid,
            FluidParameters parameters,
            DiffuseMaterialSettings settings,
            float deltaTime)
        {
            if (particles.Count == 0 || deltaTime <= 0f)
            {
                return;
            }

            SetUniforms(particles, grid, parameters, settings, deltaTime);
            BindFluid(evaluateKernel, particles, grid);
            shader.SetBuffer(evaluateKernel, SpawnWeightId, SpawnWeight);
            DispatchOver(evaluateKernel, particles.Count);

            BindFluid(spawnKernel, particles, grid);
            BindDiffuse(spawnKernel);
            shader.SetBuffer(spawnKernel, SpawnWeightId, SpawnWeight);
            DispatchOver(spawnKernel, particles.Count);

            BindFluid(advectKernel, particles, grid);
            BindDiffuse(advectKernel);
            DispatchOver(advectKernel, Capacity);
            frame++;
        }

        public void Dispose()
        {
            SpawnWeight?.Release();
            Positions?.Release();
            Velocities?.Release();
            Life?.Release();
            Kind?.Release();
            Occupied?.Release();
            SpawnWeight = null;
            Positions = null;
            Velocities = null;
            Life = null;
            Kind = null;
            Occupied = null;
        }

        void BindFluid(int kernel, ParticleSet particles, NeighborGrid grid)
        {
            shader.SetBuffer(kernel, SortedPositionsId, particles.SortedPositions);
            shader.SetBuffer(kernel, SortedVelocitiesId, particles.SortedVelocities);
            shader.SetBuffer(kernel, CellOffsetsId, grid.CellOffsets);
            shader.SetBuffer(kernel, CellCountsId, grid.CellCounts);
        }

        void BindDiffuse(int kernel)
        {
            shader.SetBuffer(kernel, DiffusePositionsId, Positions);
            shader.SetBuffer(kernel, DiffuseVelocitiesId, Velocities);
            shader.SetBuffer(kernel, DiffuseLifeId, Life);
            shader.SetBuffer(kernel, DiffuseKindId, Kind);
            shader.SetBuffer(kernel, DiffuseOccupiedId, Occupied);
        }

        void SetUniforms(
            ParticleSet particles,
            NeighborGrid grid,
            FluidParameters parameters,
            DiffuseMaterialSettings settings,
            float deltaTime)
        {
            Bounds domain = parameters.Domain;
            int3 resolution = parameters.GridResolution;
            float3 gravity = parameters.Gravity;

            shader.SetVector(DomainMinId, new Vector4(domain.min.x, domain.min.y, domain.min.z, 0f));
            shader.SetVector(DomainMinBoundsId, new Vector4(domain.min.x, domain.min.y, domain.min.z, 0f));
            shader.SetVector(DomainMaxBoundsId, new Vector4(domain.max.x, domain.max.y, domain.max.z, 0f));
            shader.SetVector(GridResolutionId, new Vector4(resolution.x, resolution.y, resolution.z, 0f));
            shader.SetFloat(CellSizeId, parameters.CellSize);
            shader.SetFloat(InverseCellSizeId, 1f / parameters.CellSize);
            shader.SetInt(CellCountId, grid.CellCount);
            shader.SetInt(ParticleCountId, particles.Count);
            shader.SetFloat(SupportRadiusId, parameters.SupportRadius);
            shader.SetFloat(KernelNormalizationId, parameters.KernelNormalization);
            shader.SetFloat(ParticleRadiusId, parameters.ParticleRadius);
            shader.SetFloat(ParticleMassId, parameters.ParticleMass);
            shader.SetFloat(DeltaTimeId, deltaTime);
            shader.SetInt(FrameId, (int)frame);
            shader.SetInt(MaxDiffuseId, Capacity);
            shader.SetFloat(PlaneDepthId, domain.center.z);
            shader.SetVector(GravityId, new Vector4(gravity.x, gravity.y, gravity.z, 0f));
            shader.SetFloat(TrappedAirMinId, settings.TrappedAirMin);
            shader.SetFloat(TrappedAirMaxId, settings.TrappedAirMax);
            shader.SetFloat(WaveCrestMinId, settings.WaveCrestMin);
            shader.SetFloat(WaveCrestMaxId, settings.WaveCrestMax);
            shader.SetFloat(EnergyMinId, settings.EnergyMin);
            shader.SetFloat(EnergyMaxId, settings.EnergyMax);
            shader.SetFloat(TrappedAirRateId, settings.TrappedAirRate);
            shader.SetFloat(WaveCrestRateId, settings.WaveCrestRate);
            shader.SetFloat(FoamLifeMinId, settings.FoamLifeMin);
            shader.SetFloat(FoamLifeMaxId, settings.FoamLifeMax);
            shader.SetFloat(BubbleBuoyancyId, settings.BubbleBuoyancy);
            shader.SetFloat(BubbleDragId, settings.BubbleDrag);
        }

        void DispatchOver(int kernel, int elementCount)
        {
            shader.Dispatch(kernel, (elementCount + ParticleGroupSize - 1) / ParticleGroupSize, 1, 1);
        }
    }

    public struct DiffuseMaterialSettings
    {
        public float TrappedAirMin;
        public float TrappedAirMax;
        public float WaveCrestMin;
        public float WaveCrestMax;
        public float EnergyMin;
        public float EnergyMax;
        public float TrappedAirRate;
        public float WaveCrestRate;
        public float FoamLifeMin;
        public float FoamLifeMax;
        public float BubbleBuoyancy;
        public float BubbleDrag;
    }
}
