using System;
using FluidSim.Core;
using FluidSim.Neighbors;
using FluidSim.Solver;
using UnityEngine;

namespace FluidSim.Rendering
{
    /// <summary>
    /// GPU 3D density field matching SebLague/Fluid-Sim <c>UpdateDensityMap</c>:
    /// each voxel is the SPH density at that world point. RayMarchingTest
    /// subtracts <c>densityOffset</c> to turn the field into a merged isosurface
    /// instead of a pile of separate particle spheres.
    /// </summary>
    public sealed class FluidDensityMap : IDisposable
    {
        static readonly int SortedPositionsId = Shader.PropertyToID("_SortedPositions");
        static readonly int CellOffsetsId = Shader.PropertyToID("_CellOffsets");
        static readonly int CellCountsId = Shader.PropertyToID("_CellCounts");
        static readonly int DensityMapId = Shader.PropertyToID("DensityMap");
        static readonly int DensityMapSizeId = Shader.PropertyToID("_DensityMapSize");
        static readonly int SupportRadiusId = Shader.PropertyToID("_SupportRadius");
        static readonly int KernelNormalizationId = Shader.PropertyToID("_KernelNormalization");
        static readonly int ParticleMassId = Shader.PropertyToID("_ParticleMass");
        static readonly int DomainCenterId = Shader.PropertyToID("_DomainCenter");
        static readonly int DomainSizeId = Shader.PropertyToID("_DomainSize");
        static readonly int DomainMinId = Shader.PropertyToID("_DomainMin");
        static readonly int GridResolutionId = Shader.PropertyToID("_GridResolution");
        static readonly int CellSizeId = Shader.PropertyToID("_CellSize");
        static readonly int InverseCellSizeId = Shader.PropertyToID("_InverseCellSize");
        static readonly int CellCountId = Shader.PropertyToID("_CellCount");
        static readonly int ParticleCountId = Shader.PropertyToID("_ParticleCount");

        const int GroupSize = 8;
        const string KernelName = "UpdateDensityTexture";

        ComputeShader shader;
        int kernel = -1;

        public RenderTexture Texture { get; private set; }

        public bool IsReady => Texture != null && shader != null && kernel >= 0;

        public void SetShader(ComputeShader compute)
        {
            if (shader == compute)
            {
                return;
            }

            shader = compute;
            kernel = shader != null ? shader.FindKernel(KernelName) : -1;
        }

        public void Update(FluidSimulation simulation, int resolution)
        {
            if (simulation == null || !simulation.IsReady || shader == null || kernel < 0)
            {
                return;
            }

            FluidParameters parameters = simulation.Parameters;
            Bounds domain = parameters.Domain;
            Vector3 size = domain.size;
            float maxAxis = Mathf.Max(size.x, Mathf.Max(size.y, size.z));
            maxAxis = Mathf.Max(maxAxis, 1e-3f);
            int res = Mathf.Clamp(resolution, 16, 160);
            int width = Mathf.Max(8, Mathf.RoundToInt(size.x / maxAxis * res));
            int height = Mathf.Max(8, Mathf.RoundToInt(size.y / maxAxis * res));
            int depth = parameters.Dimension == SimulationDimension.Two
                ? 2
                : Mathf.Max(8, Mathf.RoundToInt(size.z / maxAxis * res));

            EnsureTexture(width, height, depth);
            if (Texture == null)
            {
                return;
            }

            NeighborGrid grid = simulation.Grid;
            ParticleSet particles = simulation.Particles;
            Vector3Int gridRes = new Vector3Int(
                parameters.GridResolution.x, parameters.GridResolution.y, parameters.GridResolution.z);

            shader.SetBuffer(kernel, SortedPositionsId, particles.SortedPositions);
            shader.SetBuffer(kernel, CellOffsetsId, grid.CellOffsets);
            shader.SetBuffer(kernel, CellCountsId, grid.CellCounts);
            shader.SetTexture(kernel, DensityMapId, Texture);
            shader.SetVector(DensityMapSizeId, new Vector4(
                Texture.width, Texture.height, Texture.volumeDepth, 0f));
            shader.SetVector(DomainMinId, new Vector4(domain.min.x, domain.min.y, domain.min.z, 0f));
            shader.SetVector(GridResolutionId, new Vector4(gridRes.x, gridRes.y, gridRes.z, 0f));
            shader.SetFloat(CellSizeId, parameters.CellSize);
            shader.SetFloat(InverseCellSizeId, 1f / parameters.CellSize);
            shader.SetInt(CellCountId, grid.CellCount);
            shader.SetInt(ParticleCountId, particles.Count);
            shader.SetFloat(SupportRadiusId, parameters.SupportRadius);
            shader.SetFloat(KernelNormalizationId, parameters.KernelNormalization);
            shader.SetFloat(ParticleMassId, parameters.ParticleMass);
            shader.SetVector(DomainCenterId, domain.center);
            shader.SetVector(DomainSizeId, domain.size);

            shader.Dispatch(
                kernel,
                Mathf.CeilToInt(Texture.width / (float)GroupSize),
                Mathf.CeilToInt(Texture.height / (float)GroupSize),
                Mathf.CeilToInt(Texture.volumeDepth / (float)GroupSize));
        }

        public void Dispose()
        {
            ReleaseTexture();
            shader = null;
            kernel = -1;
        }

        void EnsureTexture(int width, int height, int depth)
        {
            if (Texture != null &&
                Texture.width == width &&
                Texture.height == height &&
                Texture.volumeDepth == depth)
            {
                return;
            }

            ReleaseTexture();
            Texture = new RenderTexture(width, height, 0, RenderTextureFormat.RHalf)
            {
                name = "FluidSim DensityMap",
                dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
                volumeDepth = depth,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                hideFlags = HideFlags.HideAndDontSave
            };
            Texture.Create();
        }

        void ReleaseTexture()
        {
            if (Texture == null)
            {
                return;
            }

            Texture.Release();
            UnityEngine.Object.DestroyImmediate(Texture);
            Texture = null;
        }
    }
}
