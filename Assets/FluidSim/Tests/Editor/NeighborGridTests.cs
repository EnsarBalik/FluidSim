using System;
using System.Collections.Generic;
using FluidSim.Core;
using FluidSim.Neighbors;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Tests
{
    /// <summary>
    /// Validates the counting-sort neighbour grid against brute force, in both dimensions.
    ///
    /// A grid that silently drops neighbours does not crash and does not look obviously wrong;
    /// it just makes the fluid behave slightly incorrectly forever. Since none of this can be
    /// stepped through in a debugger, the quadratic reference implementation is the only real
    /// ground truth available.
    ///
    /// Every test runs against both the 2D and the 3D compute shader. They are the same source
    /// with a different define, so a bug in the shared body has to show up in both, and a bug in
    /// the dimension-specific paths shows up in exactly one.
    /// </summary>
    public sealed class NeighborGridTests
    {
        const string Shader2DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid2D.compute";
        const string Shader3DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid3D.compute";

        readonly List<IDisposable> disposables = new List<IDisposable>();
        readonly List<ScriptableObject> assets = new List<ScriptableObject>();

        ComputeShader shader2D;
        ComputeShader shader3D;

        [SetUp]
        public void SetUp()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Assert.Ignore("Compute shaders are unavailable on this device.");
            }

            shader2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Shader2DPath);
            shader3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Shader3DPath);

            Assert.That(shader2D, Is.Not.Null, $"failed to load {Shader2DPath}");
            Assert.That(shader3D, Is.Not.Null, $"failed to load {Shader3DPath}");
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
        public void GridNeighbourCountsMatchBruteForce([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f);
            float3[] positions = RandomPositions(parameters, 3000, seed: 7u);

            (ParticleSet particles, NeighborGrid grid) = Build(parameters, positions);
            grid.CountNeighbors(particles, parameters);
            grid.CountNeighborsReference(particles, parameters);

            float[] fromGrid = grid.ReadNeighborCounts(particles.Count);
            float[] reference = grid.ReadReferenceNeighborCounts(particles.Count);

            for (int i = 0; i < fromGrid.Length; i++)
            {
                Assert.That(fromGrid[i], Is.EqualTo(reference[i]),
                    $"particle {i} sees {fromGrid[i]} neighbours through the grid but " +
                    $"{reference[i]} by brute force");
            }
        }

        [Test]
        public void GridNeighbourCountsMatchBruteForceWhenParticlesClump([Values(2, 3)] int dimensions)
        {
            // A clumped distribution puts many particles in very few cells, which is where an
            // off-by-one in the per-cell slice bounds shows up.
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f);

            var random = new Unity.Mathematics.Random(11u);
            var positions = new float3[2000];
            for (int i = 0; i < positions.Length; i++)
            {
                positions[i] = random.NextFloat3(new float3(-0.15f), new float3(0.15f));
            }

            (ParticleSet particles, NeighborGrid grid) = Build(parameters, positions);
            grid.CountNeighbors(particles, parameters);
            grid.CountNeighborsReference(particles, parameters);

            Assert.That(grid.ReadNeighborCounts(particles.Count),
                Is.EqualTo(grid.ReadReferenceNeighborCounts(particles.Count)));
        }

        [Test]
        public void EveryParticleAppearsExactlyOnceInSortedOrder([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f);
            float3[] positions = RandomPositions(parameters, 1500, seed: 23u);

            (ParticleSet particles, NeighborGrid grid) = Build(parameters, positions);

            uint[] sortedIndices = grid.ReadSortedIndices(particles.Count);
            var seen = new bool[particles.Count];

            foreach (uint index in sortedIndices)
            {
                Assert.That(index, Is.LessThan((uint)particles.Count), "index out of range");
                Assert.That(seen[index], Is.False, $"particle {index} was scattered twice");
                seen[index] = true;
            }

            Assert.That(Array.TrueForAll(seen, wasSeen => wasSeen), Is.True,
                "at least one particle never made it into the sorted order");
        }

        [Test]
        public void SortedOrderIsGroupedByCell([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f);
            float3[] positions = RandomPositions(parameters, 1500, seed: 41u);

            (ParticleSet particles, NeighborGrid grid) = Build(parameters, positions);

            uint[] sortedIndices = grid.ReadSortedIndices(particles.Count);
            uint[] cellIndices = grid.ReadParticleCellIndices(particles.Count);

            uint previousCell = 0u;
            for (int i = 0; i < sortedIndices.Length; i++)
            {
                uint cell = cellIndices[sortedIndices[i]];
                Assert.That(cell, Is.GreaterThanOrEqualTo(previousCell),
                    $"sorted position {i} belongs to cell {cell} after cell {previousCell}");
                previousCell = cell;
            }
        }

        [Test]
        public void CellCountsAccountForEveryParticle([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f);
            float3[] positions = RandomPositions(parameters, 1500, seed: 59u);

            (ParticleSet particles, NeighborGrid grid) = Build(parameters, positions);

            long total = 0;
            foreach (uint count in grid.ReadCellCounts())
            {
                total += count;
            }

            Assert.That(total, Is.EqualTo(particles.Count));
        }

        [Test]
        public void InteriorLatticeParticlesSeeTheDerivedNeighbourCount([Values(2, 3)] int dimensions)
        {
            // A support radius of 2.5 spacings is chosen so that no lattice shell falls on the
            // cut-off. Counting integer offsets with |k|^2 <= 6 gives 21 in two dimensions and
            // 81 in three; the nearest excluded shell is at 2.83 spacings, comfortably clear of
            // any rounding. Using a factor of 2 instead would put a whole shell exactly on the
            // boundary, where the count flickers even though the kernel is zero there.
            int expectedNeighbors = dimensions == 2 ? 21 : 81;

            const int side = 11;
            const int margin = 3;

            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2.5f, 3f);
            float spacing = parameters.ParticleSize;

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
                        float3 lattice = new float3(x - centre, y - centre,
                            dimensions == 3 ? z - centre : 0f);
                        positions[index++] = lattice * spacing;
                    }
                }
            }

            (ParticleSet particles, NeighborGrid grid) = Build(parameters, positions);
            grid.CountNeighbors(particles, parameters);

            float3[] sortedPositions = ReadSortedPositions(particles);
            float[] counts = grid.ReadNeighborCounts(particles.Count);
            float interiorExtent = (centre - margin) * spacing + 1e-4f;

            int inspected = 0;
            for (int i = 0; i < sortedPositions.Length; i++)
            {
                float3 position = sortedPositions[i];
                bool isInterior = math.abs(position.x) <= interiorExtent &&
                                  math.abs(position.y) <= interiorExtent &&
                                  (dimensions == 2 || math.abs(position.z) <= interiorExtent);
                if (!isInterior)
                {
                    continue;
                }

                inspected++;
                Assert.That(counts[i], Is.EqualTo(expectedNeighbors),
                    $"interior particle at {position} sees {counts[i]} neighbours");
            }

            Assert.That(inspected, Is.GreaterThan(0), "no interior particles were inspected");
        }

        [Test]
        public void ParticlesOutsideTheDomainAreClampedRatherThanCorrupting([Values(2, 3)] int dimensions)
        {
            // Escaping particles are a boundary-handling concern. The grid only has to survive
            // them without writing outside its buffers, and the clamp funnels many of them into
            // the same edge cell, which exercises that path hard.
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f);

            var positions = new float3[64];
            for (int i = 0; i < positions.Length; i++)
            {
                positions[i] = new float3(-50f + i, 40f, -30f);
            }

            (ParticleSet particles, NeighborGrid grid) = Build(parameters, positions);
            grid.CountNeighbors(particles, parameters);
            grid.CountNeighborsReference(particles, parameters);

            Assert.That(grid.ReadNeighborCounts(particles.Count),
                Is.EqualTo(grid.ReadReferenceNeighborCounts(particles.Count)));
        }

        (ParticleSet, NeighborGrid) Build(FluidParameters parameters, float3[] positions)
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

            ComputeShader shader = parameters.Dimension == SimulationDimension.Three
                ? shader3D
                : shader2D;

            var grid = new NeighborGrid(shader, parameters, positions.Length);
            disposables.Add(grid);
            grid.Build(particles, parameters);

            return (particles, grid);
        }

        static float3[] ReadSortedPositions(ParticleSet particles)
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

        static float3[] RandomPositions(FluidParameters parameters, int count, uint seed)
        {
            Bounds domain = parameters.Domain;
            float margin = parameters.ParticleRadius;
            float3 lower = (float3)domain.min + margin;
            float3 upper = (float3)domain.max - margin;

            var random = new Unity.Mathematics.Random(seed);
            var positions = new float3[count];
            for (int i = 0; i < count; i++)
            {
                positions[i] = random.NextFloat3(lower, upper);
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
