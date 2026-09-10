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
    public sealed class RigidBodyTests
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
        public void BoxSamplingCoversEveryFaceWithoutDuplicates([Values(2, 3)] int dimensions)
        {
            float3[] samples = RigidSurfaceSampler.SampleBox(
                new float3(0.4f, 0.3f, 0.5f), 0.1f,
                dimensions == 2 ? SimulationDimension.Two : SimulationDimension.Three);

            Assert.That(samples.Length, Is.GreaterThan(dimensions == 2 ? 7 : 20));

            var seen = new HashSet<string>();
            foreach (float3 point in samples)
            {
                string key = $"{point.x:F4},{point.y:F4},{point.z:F4}";
                Assert.That(seen.Add(key), Is.True, $"duplicate rigid sample at {point}");
            }

            Assert.That(HasPointOn(samples, p => p.x < 0f && math.abs(p.x + 0.2f) < 1e-4f));
            Assert.That(HasPointOn(samples, p => p.x > 0f && math.abs(p.x - 0.2f) < 1e-4f));
            Assert.That(HasPointOn(samples, p => p.y < 0f && math.abs(p.y + 0.15f) < 1e-4f));
            Assert.That(HasPointOn(samples, p => p.y > 0f && math.abs(p.y - 0.15f) < 1e-4f));
            if (dimensions == 3)
            {
                Assert.That(HasPointOn(samples, p => p.z < 0f && math.abs(p.z + 0.25f) < 1e-4f));
                Assert.That(HasPointOn(samples, p => p.z > 0f && math.abs(p.z - 0.25f) < 1e-4f));
            }
        }

        [Test]
        public void FluidNearRigidBoxHasHigherDensityThanAnIsolatedParticle([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 3f, Vector3.zero);
            ComputeShader gridShader = dimensions == 3 ? grid3D : grid2D;
            ComputeShader solverShader = dimensions == 3 ? solver3D : solver2D;

            var rigid = new SphRigidBody(
                gridShader, parameters, new float3(0.4f, 0.4f, 0.4f),
                new float3(0f, -0.6f, 0f), 400f, kinematic: true);
            disposables.Add(rigid);

            float3 near = new float3(0f, -0.6f + 0.2f + parameters.ParticleRadius, 0f);
            float3 far = new float3(0f, 0.8f, 0f);
            if (dimensions == 2)
            {
                near.z = 0f;
                far.z = 0f;
            }

            float nearDensity = DensityWithRigid(parameters, gridShader, solverShader, rigid, near);
            float farDensity = DensityWithRigid(parameters, gridShader, solverShader, rigid, far);

            Assert.That(nearDensity, Is.GreaterThan(farDensity * 1.5f));
            Assert.That(farDensity, Is.LessThan(parameters.RestDensity * 0.5f));
        }

        [Test]
        public void CloseFluidAndRigidParticlesRepel([Values(2, 3)] int dimensions)
        {
            FluidParameters parameters = CreateParameters(dimensions, 0.05f, 2f, 3f, Vector3.zero);
            ComputeShader gridShader = dimensions == 3 ? grid3D : grid2D;
            ComputeShader solverShader = dimensions == 3 ? solver3D : solver2D;

            var rigid = new SphRigidBody(
                gridShader, parameters, new float3(0.3f, 0.3f, 0.3f),
                float3.zero, 400f, kinematic: true);
            disposables.Add(rigid);

            float spacing = parameters.ParticleSize;
            float3[] lattice = BuildContactLattice(
                dimensions, spacing, new float3(0f, 0.15f + 1.5f * spacing, 0f));
            var fluid = new ParticleSet(lattice.Length, parameters.DimensionCount);
            disposables.Add(fluid);
            if (dimensions == 3)
            {
                fluid.Fill(lattice);
            }
            else
            {
                var flattened = new float2[lattice.Length];
                for (int i = 0; i < lattice.Length; i++)
                {
                    flattened[i] = lattice[i].xy;
                }

                fluid.Fill(flattened);
            }

            var fluidGrid = new NeighborGrid(gridShader, parameters, fluid.Count);
            disposables.Add(fluidGrid);
            fluidGrid.Build(fluid, parameters);

            var solver = new DfsphSolver(solverShader, fluid.Count, parameters.DimensionCount);
            disposables.Add(solver);
            solver.SetRigid(rigid, parameters);
            solver.ClearRigidForces();
            solver.BeginStep(fluid, fluidGrid, parameters, 0.002f, densityIterations: 2);

            float3 net = float3.zero;
            foreach (float3 force in solver.ReadRigidForces())
            {
                net += force;
            }

            Assert.That(net.y, Is.LessThan(0f), $"expected the box to be pushed down, got {net}");
        }

        [Test]
        public void FreeFallingBoxLosesHeight()
        {
            FluidParameters parameters = CreateParameters(3, 0.05f, 2f, 4f, Vector3.zero);
            var rigid = new SphRigidBody(
                grid3D, parameters, new float3(0.3f, 0.3f, 0.3f),
                new float3(0f, 0.5f, 0f), 400f);
            disposables.Add(rigid);

            float start = rigid.Position.y;
            rigid.Integrate(new float3[rigid.Count], 0.02f, new float3(0f, -9.81f, 0f), parameters.Domain);
            Assert.That(rigid.Position.y, Is.LessThan(start));
        }

        [Test]
        public void KinematicBoxIgnoresForces()
        {
            FluidParameters parameters = CreateParameters(3, 0.05f, 2f, 4f, Vector3.zero);
            var rigid = new SphRigidBody(
                grid3D, parameters, new float3(0.3f, 0.3f, 0.3f),
                new float3(0f, 0.5f, 0f), 400f, kinematic: true);
            disposables.Add(rigid);

            var forces = new float3[rigid.Count];
            for (int i = 0; i < forces.Length; i++)
            {
                forces[i] = new float3(0f, 100f, 0f);
            }

            float3 start = rigid.Position;
            rigid.Integrate(forces, 0.02f, new float3(0f, -9.81f, 0f), parameters.Domain);
            Assert.That(math.distance(rigid.Position, start), Is.EqualTo(0f));
        }

        [Test]
        public void RaycastOrientedBoxHitsTheFrontFace()
        {
            var ray = new Ray(new Vector3(0f, 0f, -2f), Vector3.forward);
            Assert.That(
                SphRigidBody.RaycastOrientedBox(
                    ray, float3.zero, quaternion.identity, new float3(1f, 1f, 1f), out float distance));
            Assert.That(distance, Is.EqualTo(1.5f).Within(1e-4f));
        }

        [Test]
        public void RaycastOrientedBoxMissesBesideTheBox()
        {
            var ray = new Ray(new Vector3(2f, 0f, -2f), Vector3.forward);
            Assert.That(
                SphRigidBody.RaycastOrientedBox(
                    ray, float3.zero, quaternion.identity, new float3(1f, 1f, 1f), out _),
                Is.False);
        }

        [Test]
        public void DragToMovesTheBoxAndSetsVelocity()
        {
            FluidParameters parameters = CreateParameters(3, 0.05f, 2f, 4f, Vector3.zero);
            var rigid = new SphRigidBody(
                grid3D, parameters, new float3(0.3f, 0.3f, 0.3f),
                new float3(0f, 0.5f, 0f), 400f);
            disposables.Add(rigid);

            rigid.DragTo(new float3(0.2f, 0.5f, 0f), 0.02f, parameters.Domain);
            Assert.That(rigid.Position.x, Is.EqualTo(0.2f).Within(1e-4f));
            Assert.That(rigid.LinearVelocity.x, Is.EqualTo(10f).Within(1e-3f));
            Assert.That((float)math.length(rigid.AngularVelocity), Is.EqualTo(0f).Within(1e-6f));
        }

        [Test]
        public void DragToStaysInsideTheDomain()
        {
            FluidParameters parameters = CreateParameters(3, 0.05f, 2f, 4f, Vector3.zero);
            var rigid = new SphRigidBody(
                grid3D, parameters, new float3(0.4f, 0.4f, 0.4f),
                float3.zero, 400f);
            disposables.Add(rigid);

            rigid.DragTo(new float3(10f, 0f, 0f), 0.02f, parameters.Domain);
            float half = 0.2f;
            float limit = 2f - half;
            Assert.That(rigid.Position.x, Is.LessThanOrEqualTo(limit + 1e-4f));
        }

        float DensityWithRigid(
            FluidParameters parameters,
            ComputeShader gridShader,
            ComputeShader solverShader,
            SphRigidBody rigid,
            float3 fluidPosition)
        {
            var fluid = new ParticleSet(1, parameters.DimensionCount);
            disposables.Add(fluid);
            if (parameters.Dimension == SimulationDimension.Three)
            {
                fluid.Fill(new[] { fluidPosition });
            }
            else
            {
                fluid.Fill(new[] { fluidPosition.xy });
            }

            var fluidGrid = new NeighborGrid(gridShader, parameters, 1);
            disposables.Add(fluidGrid);
            fluidGrid.Build(fluid, parameters);

            var solver = new DfsphSolver(solverShader, 1, parameters.DimensionCount);
            disposables.Add(solver);
            solver.SetRigid(rigid, parameters);
            solver.ComputeDensity(fluid, fluidGrid, parameters);
            return solver.ReadDensities(1)[0];
        }

        static float3[] BuildContactLattice(int dimensions, float spacing, float3 origin)
        {
            int side = 5;
            int height = 3;
            int depth = dimensions == 3 ? 5 : 1;
            var positions = new float3[side * height * depth];
            float centre = (side - 1) * 0.5f;
            float heightCentre = (height - 1) * 0.5f;
            float depthCentre = (depth - 1) * 0.5f;

            int index = 0;
            for (int z = 0; z < depth; z++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < side; x++)
                    {
                        positions[index++] = origin + new float3(
                            (x - centre) * spacing,
                            (y - heightCentre) * spacing,
                            dimensions == 3 ? (z - depthCentre) * spacing : 0f);
                    }
                }
            }

            return positions;
        }

        static bool HasPointOn(float3[] samples, Func<float3, bool> predicate)
        {
            foreach (float3 point in samples)
            {
                if (predicate(point))
                {
                    return true;
                }
            }

            return false;
        }

        FluidParameters CreateParameters(
            int dimensions, float particleRadius, float supportRadiusFactor, float domainExtent, Vector3 gravity)
        {
            var parameters = ScriptableObject.CreateInstance<FluidParameters>();
            assets.Add(parameters);

            var serialized = new SerializedObject(parameters);
            serialized.FindProperty("dimension").intValue = dimensions;
            serialized.FindProperty("particleRadius").floatValue = particleRadius;
            serialized.FindProperty("supportRadiusFactor").floatValue = supportRadiusFactor;
            serialized.FindProperty("domain").boundsValue =
                new Bounds(Vector3.zero, new Vector3(domainExtent, domainExtent, domainExtent));
            serialized.FindProperty("gravity").vector3Value = gravity;
            serialized.FindProperty("surfaceTension").floatValue = 0f;
            serialized.FindProperty("adhesion").floatValue = 0f;
            serialized.FindProperty("vorticity").floatValue = 0f;
            serialized.FindProperty("viscosityOmega").floatValue = 0f;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            return parameters;
        }
    }
}
