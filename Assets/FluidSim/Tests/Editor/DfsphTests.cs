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
    /// GPU-side checks for the divergence-free solver. The constant-density corrector is
    /// only meaningful if the density estimate itself is right, so the lattice test is the
    /// same gate as SESPH; the rest asks whether a compressed sampling actually expands.
    /// </summary>
    public sealed class DfsphTests
    {
        const string Grid2DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid2D.compute";
        const string Grid3DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid3D.compute";
        const string Solver2DPath = "Assets/FluidSim/Shaders/Compute/Dfsph2D.compute";
        const string Solver3DPath = "Assets/FluidSim/Shaders/Compute/Dfsph3D.compute";

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
            Assert.That(solver2D, Is.Not.Null, "Dfsph2D.compute is missing or failed to import.");
            Assert.That(solver3D, Is.Not.Null, "Dfsph3D.compute is missing or failed to import.");
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

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
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
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero);
            var positions = new[] { float3.zero };

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            grid.Build(particles, parameters);
            solver.BeginStep(particles, grid, parameters, 0.002f, densityIterations: 1);

            Assert.That(solver.ReadPressures(1)[0], Is.EqualTo(0f));
            Assert.That(solver.ReadDensities(1)[0], Is.LessThan(parameters.RestDensity * 0.5f));
            Assert.That(solver.ReadStiffnessFactors(1)[0], Is.EqualTo(0f));
        }

        [Test]
        public void NearlyCoincidentParticlesStayBounded([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero);
            var positions = new[]
            {
                float3.zero,
                new float3(1e-4f, 0f, 0f)
            };

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);

            for (int i = 0; i < 25; i++)
            {
                grid.Build(particles, parameters);
                solver.Advance(particles, grid, parameters, 0.002f, () => grid.Build(particles, parameters));
            }

            grid.Build(particles, parameters);
            float3[] after = ReadSorted(particles);
            Bounds domain = parameters.Domain;
            float limit = domain.extents.x - parameters.ParticleRadius;

            foreach (float3 point in after)
            {
                Assert.That(math.abs(point.x), Is.LessThan(limit), $"exploded to {point}");
                Assert.That(math.abs(point.y), Is.LessThan(limit), $"exploded to {point}");
                if (dimensions == 3)
                {
                    Assert.That(math.abs(point.z), Is.LessThan(limit), $"exploded to {point}");
                }
            }
        }

        [Test]
        public void SparseNeighborhoodDoesNotExplode([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero);
            float spacing = parameters.ParticleSize;
            var positions = new[]
            {
                float3.zero,
                new float3(spacing, 0f, 0f),
                new float3(0f, spacing, 0f),
                new float3(spacing, spacing, 0f)
            };

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);

            for (int i = 0; i < 40; i++)
            {
                grid.Build(particles, parameters);
                solver.Advance(particles, grid, parameters, 0.002f, () => grid.Build(particles, parameters));
            }

            grid.Build(particles, parameters);
            float3[] after = ReadSorted(particles);
            Bounds domain = parameters.Domain;
            float limit = domain.extents.x - parameters.ParticleRadius;
            float[] factors = solver.ReadStiffnessFactors(particles.Count);

            foreach (float factor in factors)
            {
                Assert.That(factor, Is.EqualTo(0f).Within(1e-8f));
            }

            foreach (float3 point in after)
            {
                Assert.That(math.abs(point.x), Is.LessThan(limit), $"exploded to {point}");
                Assert.That(math.abs(point.y), Is.LessThan(limit), $"exploded to {point}");
                if (dimensions == 3)
                {
                    Assert.That(math.abs(point.z), Is.LessThan(limit), $"exploded to {point}");
                }
            }
        }

        [Test]
        public void NeighborFadeIsZeroForSprayAndOneForAFullSheet()
        {
            float Fade(int count, int min, int full) =>
                Mathf.Clamp01((count - min) / (float)(full - min));

            Assert.That(Fade(4, 8, 22), Is.EqualTo(0f));
            Assert.That(Fade(8, 8, 22), Is.EqualTo(0f));
            Assert.That(Fade(15, 8, 22), Is.GreaterThan(0.4f).And.LessThan(0.6f));
            Assert.That(Fade(22, 8, 22), Is.EqualTo(1f));
            Assert.That(Fade(40, 8, 22), Is.EqualTo(1f));
        }

        [Test]
        public void InteriorStiffnessFactorIsPositiveAndFinite([Values(2, 3)] int dimensions)
        {
            const int side = 7;
            const int margin = 2;

            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero);
            float spacing = parameters.ParticleSize;
            float3[] positions = BuildLattice(dimensions, side, spacing);

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            grid.Build(particles, parameters);
            solver.BeginStep(particles, grid, parameters, 0.002f, densityIterations: 1);

            float3[] sorted = ReadSorted(particles);
            float[] factors = solver.ReadStiffnessFactors(particles.Count);
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
                Assert.That(factors[i], Is.GreaterThan(0f));
                Assert.That(float.IsFinite(factors[i]), Is.True, $"k at {sorted[i]} is {factors[i]}");
            }

            Assert.That(inspected, Is.GreaterThan(0));
        }

        [Test]
        public void CompressedLatticeExpandsTowardRestSpacing([Values(2, 3)] int dimensions)
        {
            int side = dimensions == 3 ? 5 : 7;
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero);
            float restSpacing = parameters.ParticleSize;
            float packedSpacing = restSpacing * 0.65f;
            float3[] positions = BuildLattice(dimensions, side, packedSpacing);

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            float extentBefore = BoundingExtent(positions);

            for (int i = 0; i < 24; i++)
            {
                grid.Build(particles, parameters);
                solver.Advance(particles, grid, parameters, 0.002f, () => grid.Build(particles, parameters));
            }

            grid.Build(particles, parameters);
            float extentAfter = BoundingExtent(ReadSorted(particles));

            Assert.That(extentAfter, Is.GreaterThan(extentBefore * 1.05f),
                $"compressed cloud did not expand ({extentBefore} → {extentAfter})");
        }

        [Test]
        public void BoxCollisionKeepsParticlesInsideTheDomain()
        {
            FluidParameters parameters = CreateParameters(3, 0.05f, 2f, 2f);
            var positions = new[] { new float3(0f, 0.9f, 0f) };

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);

            for (int i = 0; i < 40; i++)
            {
                grid.Build(particles, parameters);
                solver.Advance(particles, grid, parameters, 0.01f, () => grid.Build(particles, parameters));
            }

            float3[] after = particles.ReadSortedPositions3D();
            Bounds domain = parameters.Domain;
            float limit = domain.max.y - parameters.ParticleRadius + 1e-4f;

            Assert.That(after[0].y, Is.LessThanOrEqualTo(limit));
            Assert.That(after[0].y, Is.GreaterThanOrEqualTo(domain.min.y + parameters.ParticleRadius - 1e-4f));
        }

        [Test]
        public void SurfaceParticleHasLargerColourFieldThanInterior([Values(2, 3)] int dimensions)
        {
            const int side = 9;
            const int margin = 3;

            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 3f, surfaceTension: 0.1f);
            float spacing = parameters.ParticleSize;
            float3[] positions = BuildLattice(dimensions, side, spacing);

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            grid.Build(particles, parameters);
            solver.ComputeDensity(particles, grid, parameters);
            solver.ComputeSurfaceNormals(particles, grid, parameters);

            float3[] sorted = ReadSorted(particles);
            float[] normals = solver.ReadSurfaceNormalLengths(particles.Count);
            float interiorExtent = ((side - 1) * 0.5f - margin) * spacing + 1e-4f;
            float surfaceBand = ((side - 1) * 0.5f) * spacing + 1e-4f;

            float interiorMean = 0f;
            int interiorCount = 0;
            float surfaceMean = 0f;
            int surfaceCount = 0;
            for (int i = 0; i < sorted.Length; i++)
            {
                bool isInterior = math.abs(sorted[i].x) <= interiorExtent &&
                                  math.abs(sorted[i].y) <= interiorExtent &&
                                  (dimensions == 2 || math.abs(sorted[i].z) <= interiorExtent);
                bool isSurface = math.abs(sorted[i].x) >= surfaceBand - spacing * 0.1f ||
                                 math.abs(sorted[i].y) >= surfaceBand - spacing * 0.1f ||
                                 (dimensions == 3 && math.abs(sorted[i].z) >= surfaceBand - spacing * 0.1f);

                if (isInterior)
                {
                    interiorMean += normals[i];
                    interiorCount++;
                }
                else if (isSurface)
                {
                    surfaceMean += normals[i];
                    surfaceCount++;
                }
            }

            Assert.That(interiorCount, Is.GreaterThan(0));
            Assert.That(surfaceCount, Is.GreaterThan(0));
            interiorMean /= interiorCount;
            surfaceMean /= surfaceCount;
            Assert.That(surfaceMean, Is.GreaterThan(interiorMean * 1.5f),
                $"surface |n| {surfaceMean} should exceed interior {interiorMean}");
        }

        [Test]
        public void MidRangePairIsPulledTogetherByCohesion([Values(2, 3)] int dimensions)
        {
            // Two isolated particles have a much larger self-density in 2D (mass is
            // ρ0 h~²), so the Akinci symmetry factor 2ρ0/(ρi+ρj) is smaller and
            // the 1e-3 viscosity floor eats most of a short run. Give 2D more
            // time rather than pretending the 3D coefficient maps 1:1.
            FluidParameters parameters = CreateParameters(
                dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero, surfaceTension: 1.0f);
            float gap = parameters.SupportRadius * 0.72f;
            var positions = new[]
            {
                new float3(-0.5f * gap, 0f, 0f),
                new float3(0.5f * gap, 0f, 0f)
            };

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);

            int steps = dimensions == 2 ? 80 : 40;
            for (int i = 0; i < steps; i++)
            {
                grid.Build(particles, parameters);
                solver.Advance(particles, grid, parameters, 0.002f, () => grid.Build(particles, parameters));
            }

            grid.Build(particles, parameters);
            float3[] after = ReadSorted(particles);
            float distance = math.distance(after[0], after[1]);
            Assert.That(distance, Is.LessThan(gap * 0.97f),
                $"cohesion did not pull the pair together ({gap} → {distance})");
        }

        [Test]
        public void RestingLatticeKeepsZeroMicrorotation([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(
                dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero, vorticity: 0.2f, viscosityOmega: 0.1f);
            float3[] positions = BuildLattice(dimensions, 5, parameters.ParticleSize);

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            grid.Build(particles, parameters);
            solver.BeginStep(particles, grid, parameters, 0.002f, densityIterations: 1);

            float[] speeds = solver.ReadAngularSpeeds(particles.Count);
            foreach (float speed in speeds)
            {
                Assert.That(speed, Is.LessThan(1e-3f), $"resting particle spun up to {speed}");
            }
        }

        [Test]
        public void RecedingPairLosesRelativeSpeedWithImplicitViscosity([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(
                dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero,
                kinematicViscosity: 10f, implicitViscosityIterations: 10);
            Assert.That(parameters.ImplicitViscosityIterations, Is.EqualTo(10));
            Assert.That(parameters.KinematicViscosity, Is.EqualTo(10f));
            float gap = parameters.SupportRadius * 0.5f;
            var positions = new[]
            {
                new float3(-0.5f * gap, 0f, 0f),
                new float3(0.5f * gap, 0f, 0f)
            };

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            // Brookshaw / Weiler only sees the radial strain (v_ij · r_ij).
            // A perpendicular shear pair is invisible to that operator — the
            // micropolar test uses exactly that setup on purpose. Receding
            // along the separation axis is the mode viscosity can damp, and
            // DFSPH pressure cannot fake it: we never attract, only push.
            if (dimensions == 3)
            {
                particles.Velocities.SetData(new[] { new float3(-1.5f, 0f, 0f), new float3(1.5f, 0f, 0f) });
            }
            else
            {
                particles.Velocities.SetData(new[] { new float2(-1.5f, 0f), new float2(1.5f, 0f) });
            }

            grid.Build(particles, parameters);
            solver.BeginStep(particles, grid, parameters, 0.002f, densityIterations: 1);

            float relative = RelativeSpeed(particles);
            Assert.That(relative, Is.LessThan(2.7f),
                $"implicit viscosity should damp a receding pair, but relative speed is {relative}");
        }

        [Test]
        public void RestingLatticeStaysQuietWithImplicitViscosity([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(
                dimensions, 0.05f, 2f, 3f, gravity: Vector3.zero,
                kinematicViscosity: 10f, implicitViscosityIterations: 8);
            float3[] positions = BuildLattice(dimensions, 5, parameters.ParticleSize);

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            grid.Build(particles, parameters);
            solver.BeginStep(particles, grid, parameters, 0.002f, densityIterations: 1);

            Assert.That(RelativeSpeedFromRest(particles), Is.LessThan(0.05f));
        }

        [Test]
        public void ShearingPairGainsMicrorotation([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(
                dimensions, 0.05f, 2f, 2f, gravity: Vector3.zero, vorticity: 0.4f, viscosityOmega: 0.05f);
            float gap = parameters.SupportRadius * 0.5f;
            var positions = new[]
            {
                new float3(-0.5f * gap, 0f, 0f),
                new float3(0.5f * gap, 0f, 0f)
            };

            (ParticleSet particles, NeighborGrid grid, DfsphSolver solver) = Build(parameters, positions);
            if (dimensions == 3)
            {
                particles.Velocities.SetData(new[] { new float3(0f, 1.5f, 0f), new float3(0f, -1.5f, 0f) });
            }
            else
            {
                particles.Velocities.SetData(new[] { new float2(0f, 1.5f), new float2(0f, -1.5f) });
            }

            grid.Build(particles, parameters);
            solver.BeginStep(particles, grid, parameters, 0.002f, densityIterations: 1);

            float[] speeds = solver.ReadAngularSpeeds(particles.Count);
            Assert.That(speeds[0] + speeds[1], Is.GreaterThan(1e-4f),
                "a shearing pair should spin up a microrotation");
        }

        (ParticleSet, NeighborGrid, DfsphSolver) Build(FluidParameters parameters, float3[] positions)
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

            var solver = new DfsphSolver(solverShader, positions.Length, parameters.DimensionCount);
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

        static float RelativeSpeed(ParticleSet particles)
        {
            if (particles.Dimensions == 3)
            {
                var velocities = new float3[particles.Count];
                particles.Velocities.GetData(velocities, 0, 0, particles.Count);
                return math.length(velocities[0] - velocities[1]);
            }

            var flat = new float2[particles.Count];
            particles.Velocities.GetData(flat, 0, 0, particles.Count);
            return math.length(flat[0] - flat[1]);
        }

        static float RelativeSpeedFromRest(ParticleSet particles)
        {
            float maxSpeed = 0f;
            if (particles.Dimensions == 3)
            {
                var velocities = new float3[particles.Count];
                particles.Velocities.GetData(velocities, 0, 0, particles.Count);
                for (int i = 0; i < velocities.Length; i++)
                {
                    maxSpeed = math.max(maxSpeed, math.length(velocities[i]));
                }

                return maxSpeed;
            }

            var flat = new float2[particles.Count];
            particles.Velocities.GetData(flat, 0, 0, particles.Count);
            for (int i = 0; i < flat.Length; i++)
            {
                maxSpeed = math.max(maxSpeed, math.length(flat[i]));
            }

            return maxSpeed;
        }

        static float BoundingExtent(float3[] positions)
        {
            float3 minimum = positions[0];
            float3 maximum = positions[0];
            for (int i = 1; i < positions.Length; i++)
            {
                minimum = math.min(minimum, positions[i]);
                maximum = math.max(maximum, positions[i]);
            }

            return math.length(maximum - minimum);
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
            int dimensions,
            float particleRadius,
            float supportRadiusFactor,
            float domainExtent,
            Vector3? gravity = null,
            float surfaceTension = 0f,
            float adhesion = 0f,
            float vorticity = 0f,
            float viscosityOmega = 0f,
            float kinematicViscosity = 1e-6f,
            int implicitViscosityIterations = 0)
        {
            var parameters = ScriptableObject.CreateInstance<FluidParameters>();
            assets.Add(parameters);

            var serialized = new SerializedObject(parameters);
            serialized.FindProperty("dimension").intValue = dimensions;
            serialized.FindProperty("particleRadius").floatValue = particleRadius;
            serialized.FindProperty("supportRadiusFactor").floatValue = supportRadiusFactor;
            serialized.FindProperty("domain").boundsValue =
                new Bounds(Vector3.zero, new Vector3(domainExtent, domainExtent, domainExtent));
            serialized.FindProperty("surfaceTension").floatValue = surfaceTension;
            serialized.FindProperty("adhesion").floatValue = adhesion;
            serialized.FindProperty("vorticity").floatValue = vorticity;
            serialized.FindProperty("viscosityOmega").floatValue = viscosityOmega;
            serialized.FindProperty("kinematicViscosity").floatValue = kinematicViscosity;
            serialized.FindProperty("implicitViscosityIterations").intValue = implicitViscosityIterations;
            if (gravity.HasValue)
            {
                serialized.FindProperty("gravity").vector3Value = gravity.Value;
            }

            serialized.ApplyModifiedPropertiesWithoutUndo();

            return parameters;
        }
    }
}
