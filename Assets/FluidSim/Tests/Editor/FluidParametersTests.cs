using FluidSim.Core;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Tests
{
    /// <summary>
    /// Locks down the derivation chain that every other part of the simulation reads from.
    /// </summary>
    public sealed class FluidParametersTests
    {
        FluidParameters parameters;

        [SetUp]
        public void SetUp()
        {
            parameters = ScriptableObject.CreateInstance<FluidParameters>();
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(parameters);
        }

        [Test]
        public void ParticleMassFillsTheVolumeAParticleOccupies()
        {
            float expected = parameters.Dimension == SimulationDimension.Two
                ? parameters.RestDensity * parameters.ParticleSize * parameters.ParticleSize
                : parameters.RestDensity * parameters.ParticleSize * parameters.ParticleSize *
                  parameters.ParticleSize;

            Assert.That(parameters.ParticleMass, Is.EqualTo(expected).Within(expected * 1e-6f));
        }

        [Test]
        public void SupportRadiusYieldsAWorkableNeighbourhood()
        {
            // The tutorial's heuristic targets 30-40 neighbours in 3D, which is the same
            // support-to-spacing ratio that gives roughly 13 in 2D.
            float expected = parameters.Dimension == SimulationDimension.Two ? 12.57f : 33.51f;

            Assert.That(parameters.ExpectedNeighborCount, Is.EqualTo(expected).Within(0.1f));
        }

        [Test]
        public void GridCellsAreSizedToTheSupportRadius()
        {
            // A one-ring query is only correct when a cell is at least as wide as the support.
            Assert.That(parameters.CellSize, Is.EqualTo(parameters.SupportRadius));

            int3 resolution = parameters.GridResolution;
            Assert.That(math.all(resolution >= 1), Is.True, "every axis needs at least one cell");

            // Small slack because the cell size is not exactly representable in binary.
            Vector3 size = parameters.Domain.size;
            Assert.That(resolution.x * parameters.CellSize,
                Is.GreaterThanOrEqualTo(size.x - 1e-4f), "grid must cover the domain on x");
            Assert.That(resolution.y * parameters.CellSize,
                Is.GreaterThanOrEqualTo(size.y - 1e-4f), "grid must cover the domain on y");
            if (parameters.Dimension == SimulationDimension.Three)
            {
                Assert.That(resolution.z * parameters.CellSize,
                    Is.GreaterThanOrEqualTo(size.z - 1e-4f), "grid must cover the domain on z");
            }

            Assert.That(parameters.CellCount,
                Is.EqualTo(resolution.x * resolution.y * resolution.z));
        }

        [Test]
        public void TwoDimensionalGridCollapsesTheDepthAxis()
        {
            var serialized = new SerializedObject(parameters);
            serialized.FindProperty("dimension").intValue = (int)SimulationDimension.Two;
            serialized.ApplyModifiedPropertiesWithoutUndo();

            Assert.That(parameters.GridResolution.z, Is.EqualTo(1));
        }

        [Test]
        public void TimeStepScalesInverselyWithSpeedInsideTheClampRange()
        {
            // Well above the speed at which the CFL bound is still clipped by maxTimeStep.
            float slow = parameters.TimeStepFor(100f);
            float fast = parameters.TimeStepFor(200f);

            Assert.That(slow / fast, Is.EqualTo(2f).Within(1e-3f));
            Assert.That(slow, Is.EqualTo(parameters.CflFactor * parameters.ParticleSize / 100f)
                .Within(1e-9f));
        }

        [Test]
        public void SurfaceTensionDefaultsAreNonNegative()
        {
            Assert.That(parameters.SurfaceTension, Is.GreaterThanOrEqualTo(0f));
            Assert.That(parameters.Adhesion, Is.GreaterThanOrEqualTo(0f));
        }

        [Test]
        public void MicropolarDefaultsAreNonNegative()
        {
            Assert.That(parameters.Vorticity, Is.GreaterThanOrEqualTo(0f));
            Assert.That(parameters.ViscosityOmega, Is.GreaterThanOrEqualTo(0f));
            Assert.That(parameters.InertiaInverse, Is.GreaterThan(0f));
        }

        [Test]
        public void ImplicitViscosityDefaultsToExplicitBrookshaw()
        {
            Assert.That(parameters.ImplicitViscosityIterations, Is.EqualTo(0));
        }

        [Test]
        public void RuntimeDomainOverrideDoesNotChangeTheSerializedAsset()
        {
            Bounds authored = parameters.AuthoredDomain;
            parameters.SetRuntimeDomain(new Bounds(Vector3.zero, new Vector3(2f, 2f, 2f)));

            Assert.That(parameters.Domain.size.x, Is.EqualTo(2f).Within(1e-4f));
            Assert.That(parameters.AuthoredDomain.size, Is.EqualTo(authored.size));

            parameters.ClearRuntimeDomain();
            Assert.That(parameters.Domain.size, Is.EqualTo(authored.size));
        }

        [Test]
        public void TimeStepIsClampedForVanishingAndExtremeSpeeds()
        {
            float atRest = parameters.TimeStepFor(0f);
            float absurd = parameters.TimeStepFor(1e6f);

            Assert.That(atRest, Is.GreaterThan(0f));
            Assert.That(absurd, Is.GreaterThan(0f));
            Assert.That(absurd, Is.LessThan(atRest));
        }
    }
}
