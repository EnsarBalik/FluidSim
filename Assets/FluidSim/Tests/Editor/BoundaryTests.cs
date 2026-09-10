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
    public sealed class BoundaryTests
    {
        const string Grid3DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid3D.compute";
        const string Solver3DPath = "Assets/FluidSim/Shaders/Compute/Sesph3D.compute";

        readonly List<IDisposable> disposables = new List<IDisposable>();
        readonly List<ScriptableObject> assets = new List<ScriptableObject>();

        ComputeShader gridShader;
        ComputeShader solverShader;

        [SetUp]
        public void SetUp()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Assert.Ignore("Compute shaders are unavailable on this device.");
            }

            gridShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(Grid3DPath);
            solverShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(Solver3DPath);
            Assert.That(gridShader, Is.Not.Null);
            Assert.That(solverShader, Is.Not.Null);
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
        public void BoxSamplingCoversEveryFaceWithoutDuplicates()
        {
            FluidParameters parameters = CreateParameters(3, 0.2f, 2f, 2f);
            float3[] samples = DomainBoundarySampler.Sample(parameters);

            Assert.That(samples.Length, Is.GreaterThan(8));

            var seen = new HashSet<string>();
            foreach (float3 point in samples)
            {
                string key = $"{point.x:F4},{point.y:F4},{point.z:F4}";
                Assert.That(seen.Add(key), Is.True, $"duplicate boundary sample at {point}");
            }

            Bounds domain = parameters.Domain;
            Assert.That(HasPointOn(samples, p => math.abs(p.x - domain.min.x) < 1e-4f));
            Assert.That(HasPointOn(samples, p => math.abs(p.x - domain.max.x) < 1e-4f));
            Assert.That(HasPointOn(samples, p => math.abs(p.y - domain.min.y) < 1e-4f));
            Assert.That(HasPointOn(samples, p => math.abs(p.y - domain.max.y) < 1e-4f));
            Assert.That(HasPointOn(samples, p => math.abs(p.z - domain.min.z) < 1e-4f));
            Assert.That(HasPointOn(samples, p => math.abs(p.z - domain.max.z) < 1e-4f));
        }

        [Test]
        public void FluidNearAWallHasHigherDensityThanAnIsolatedParticle()
        {
            FluidParameters parameters = CreateParameters(3, 0.05f, 2f, 2f);

            var isolated = new ParticleSet(1, 3);
            disposables.Add(isolated);
            isolated.Fill(new[] { new float3(0f, 0f, 0f) });

            var againstFloor = new ParticleSet(1, 3);
            disposables.Add(againstFloor);
            againstFloor.Fill(new[]
            {
                new float3(0f, parameters.Domain.min.y + parameters.ParticleRadius, 0f)
            });

            float isolatedDensity = DensityWithBoundary(parameters, isolated);
            float floorDensity = DensityWithBoundary(parameters, againstFloor);

            Assert.That(floorDensity, Is.GreaterThan(isolatedDensity * 1.5f));
            Assert.That(isolatedDensity, Is.LessThan(parameters.RestDensity * 0.5f));
        }

        float DensityWithBoundary(FluidParameters parameters, ParticleSet fluid)
        {
            var fluidGrid = new NeighborGrid(gridShader, parameters, fluid.Count);
            disposables.Add(fluidGrid);
            fluidGrid.Build(fluid, parameters);

            float3[] samples = DomainBoundarySampler.Sample(parameters);
            var boundary = new ParticleSet(samples.Length, 3);
            disposables.Add(boundary);
            boundary.Fill(samples);

            var boundaryGrid = new NeighborGrid(gridShader, parameters, samples.Length);
            disposables.Add(boundaryGrid);
            boundaryGrid.Build(boundary, parameters);

            var solver = new SesphSolver(solverShader, fluid.Count, 3);
            disposables.Add(solver);
            solver.SetBoundary(boundary, boundaryGrid, parameters);
            solver.ComputeDensity(fluid, fluidGrid, parameters);

            return solver.ReadDensities(1)[0];
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
