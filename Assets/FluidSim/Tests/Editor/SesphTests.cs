using System;
using System.Collections.Generic;
using FluidSim.Core;
using FluidSim.Neighbors;
using FluidSim.Solver;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Tests
{
    /// <summary>
    /// GPU-side checks for the state-equation solver. A lattice that does not reconstruct
    /// the rest density means every pressure force downstream is starting from the wrong
    /// number, and no amount of stiffness tuning will hide it.
    /// </summary>
    public sealed class SesphTests
    {
        const string Grid2DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid2D.compute";
        const string Grid3DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid3D.compute";
        const string Solver2DPath = "Assets/FluidSim/Shaders/Compute/Sesph2D.compute";
        const string Solver3DPath = "Assets/FluidSim/Shaders/Compute/Sesph3D.compute";

        readonly List<IDisposable> disposables = new List<IDisposable>();
        readonly List<ScriptableObject> assets = new List<ScriptableObject>();

        ComputeShader grid2D;
        ComputeShader grid3D;
        ComputeShader solver2D;
        ComputeShader solver3D;

        [SetUp]
        public void SetUp()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Assert.Ignore("Compute shaders are unavailable on this device.");
            }

            grid2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Grid2DPath);
            grid3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Grid3DPath);
            solver2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Solver2DPath);
            solver3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Solver3DPath);

            Assert.That(grid2D, Is.Not.Null);
            Assert.That(grid3D, Is.Not.Null);
            Assert.That(solver2D, Is.Not.Null);
            Assert.That(solver3D, Is.Not.Null);
        }

        [TearDown]
        public void TearDown()
        {
            foreach (IDisposable disposable in disposables)
            {
                disposable.Dispose();
            }

            disposables.Clear();

            foreach (ScriptableObject asset in assets)
            {
                UnityEngine.Object.DestroyImmediate(asset);
            }

            assets.Clear();
        }

        [Test]
        public void InteriorLatticeDensityMatchesRestDensity([Values(2, 3)] int dimensions)
        {
            const int side = 11;
            const int margin = 3;

            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 3f);
            float spacing = parameters.ParticleSize;
            float3[] positions = BuildLattice(dimensions, side, spacing);

            (ParticleSet particles, NeighborGrid grid, SesphSolver solver) = Build(parameters, positions);
            grid.Build(particles, parameters);
            solver.ComputeDensity(particles, grid, parameters);

            float3[] sorted = ReadSorted(particles);
            float[] densities = solver.ReadDensities(particles.Count);
            float interiorExtent = ((side - 1) * 0.5f - margin) * spacing + 1e-4f;

            int inspected = 0;
            for (int i = 0; i < sorted.Length; i++)
            {
                bool isInterior = math.abs(sorted[i].x) <= interiorExtent &&
                                  math.abs(sorted[i].y) <= interiorExtent &&
                                  (dimensions == 2 || math.abs(sorted[i].z) <= interiorExtent);
                if (!isInterior)
                {
                    continue;
                }

                inspected++;
                Assert.That(densities[i],
                    Is.EqualTo(parameters.RestDensity).Within(parameters.RestDensity * 0.02f),
                    $"interior particle at {sorted[i]} has density {densities[i]}");
            }

            Assert.That(inspected, Is.GreaterThan(0));
        }

        [Test]
        public void IsolatedParticlePressureIsClampedToZero([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f);
            var positions = new[] { float3.zero };

            (ParticleSet particles, NeighborGrid grid, SesphSolver solver) = Build(parameters, positions);
            grid.Build(particles, parameters);
            solver.ComputeDensity(particles, grid, parameters);

            // A lone particle only sees its own kernel self-term, so density is far below
            // rest density. Negative pressure is clamped so free-surface particles are not
            // sucked inward by the particle-deficiency artefact.
            Assert.That(solver.ReadPressures(1)[0], Is.EqualTo(0f));
            Assert.That(solver.ReadDensities(1)[0], Is.LessThan(parameters.RestDensity * 0.5f));
        }

        [Test]
        public void BoxCollisionKeepsParticlesInsideTheDomain()
        {
            FluidParameters parameters = CreateParameters(3, 0.05f, 2f, 2f);
            var positions = new[] { new float3(0f, 0.9f, 0f) };

            (ParticleSet particles, NeighborGrid grid, SesphSolver solver) = Build(parameters, positions);

            for (int i = 0; i < 40; i++)
            {
                grid.Build(particles, parameters);
                solver.Advance(particles, grid, parameters, 0.01f);
            }

            float3[] after = particles.ReadSortedPositions3D();
            Bounds domain = parameters.Domain;
            float limit = domain.max.y - parameters.ParticleRadius + 1e-4f;

            Assert.That(after[0].y, Is.LessThanOrEqualTo(limit));
            Assert.That(after[0].y, Is.GreaterThanOrEqualTo(domain.min.y + parameters.ParticleRadius - 1e-4f));
        }

        (ParticleSet, NeighborGrid, SesphSolver) Build(FluidParameters parameters, float3[] positions)
        {
            var particles = new ParticleSet(positions.Length, parameters.DimensionCount);
            disposables.Add(particles);

            if (parameters.Dimension == SimulationDimension.Three)
            {
                particles.Fill(positions);
            }
            else
            {
                var flattened = new float2[positions.Length];
                for (int i = 0; i < positions.Length; i++)
                {
                    flattened[i] = positions[i].xy;
                }

                particles.Fill(flattened);
            }

            ComputeShader gridShader = parameters.Dimension == SimulationDimension.Three ? grid3D : grid2D;
            ComputeShader solverShader = parameters.Dimension == SimulationDimension.Three ? solver3D : solver2D;

            var grid = new NeighborGrid(gridShader, parameters, positions.Length);
            disposables.Add(grid);

            var solver = new SesphSolver(solverShader, positions.Length, parameters.DimensionCount);
            disposables.Add(solver);

            return (particles, grid, solver);
        }

        static float3[] ReadSorted(ParticleSet particles)
        {
            if (particles.Dimensions == 3)
            {
                return particles.ReadSortedPositions3D();
            }

            float2[] flat = particles.ReadSortedPositions2D();
            var promoted = new float3[flat.Length];
            for (int i = 0; i < flat.Length; i++)
            {
                promoted[i] = new float3(flat[i], 0f);
            }

            return promoted;
        }

        static float3[] BuildLattice(int dimensions, int side, float spacing)
        {
            int depth = dimensions == 3 ? side : 1;
            var positions = new float3[side * side * depth];
            float centre = (side - 1) * 0.5f;

            int index = 0;
            for (int z = 0; z < depth; z++)
            {
                for (int y = 0; y < side; y++)
                {
                    for (int x = 0; x < side; x++)
                    {
                        positions[index++] = new float3(
                            (x - centre) * spacing,
                            (y - centre) * spacing,
                            dimensions == 3 ? (z - centre) * spacing : 0f);
                    }
                }
            }

            return positions;
        }

        FluidParameters CreateParameters(
            int dimensions, float particleRadius, float supportRadiusFactor, float domainExtent)
        {
            var parameters = ScriptableObject.CreateInstance<FluidParameters>();
            assets.Add(parameters);

            var serialized = new SerializedObject(parameters);
            serialized.FindProperty("dimension").intValue = dimensions;
            serialized.FindProperty("particleRadius").floatValue = particleRadius;
            serialized.FindProperty("supportRadiusFactor").floatValue = supportRadiusFactor;
            serialized.FindProperty("domain").boundsValue =
                new Bounds(Vector3.zero, new Vector3(domainExtent, domainExtent, domainExtent));
            serialized.ApplyModifiedPropertiesWithoutUndo();

            return parameters;
        }
    }
}
