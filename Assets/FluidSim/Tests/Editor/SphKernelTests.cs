using System;
using FluidSim.Core;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Tests
{
    /// <summary>
    /// Validates the cubic spline kernel before any solver is built on top of it.
    /// A wrong normalization or a sign slip in the gradient is nearly impossible to
    /// diagnose once it only shows up as "the fluid explodes" on the GPU.
    /// </summary>
    public sealed class SphKernelTests
    {
        const string ProbeShaderPath = "Assets/FluidSim/Shaders/Compute/SphKernelProbe.compute";

        const float SupportRadius = 0.1f;
        const float RestDensity = 1000f;

        static float Normalization2D => SphKernel.Normalization(2, SupportRadius);

        static float Normalization3D => SphKernel.Normalization(3, SupportRadius);

        [Test]
        public void ProfilePeaksAtCentreAndVanishesOutsideSupport()
        {
            Assert.That(SphKernel.Profile(0f), Is.EqualTo(1f).Within(1e-6f));
            Assert.That(SphKernel.Profile(1f), Is.Zero);
            Assert.That(SphKernel.Profile(1.5f), Is.Zero);
            Assert.That(SphKernel.ProfileDerivative(1f), Is.Zero);
        }

        [Test]
        public void ProfileMatchesAnalyticValuesWhereTheTwoSegmentsMeet()
        {
            // Both branches of Eq. (4) must agree at q = 1/2, otherwise forces jump
            // discontinuously as particles drift across half the support radius.
            Assert.That(SphKernel.Profile(0.5f), Is.EqualTo(0.25f).Within(1e-6f));
            Assert.That(SphKernel.ProfileDerivative(0.5f), Is.EqualTo(-1.5f).Within(1e-6f));
        }

        [Test]
        public void ProfileDecreasesMonotonicallyAcrossSupport()
        {
            float previous = SphKernel.Profile(0f);
            for (int step = 1; step <= 200; step++)
            {
                float current = SphKernel.Profile(step / 200f);
                Assert.That(current, Is.LessThanOrEqualTo(previous), $"not monotonic at q = {step / 200f}");
                previous = current;
            }
        }

        [Test]
        public void NormalizationMakesKernelIntegrateToUnity([Values(1, 2, 3)] int dimension)
        {
            const int steps = 200000;
            float normalization = SphKernel.Normalization(dimension, SupportRadius);
            double stepSize = SupportRadius / (double)steps;
            double integral = 0.0;

            for (int i = 0; i < steps; i++)
            {
                double radius = (i + 0.5) * stepSize;
                double value = SphKernel.Evaluate((float)radius, SupportRadius, normalization);

                // Measure of the shell at this radius: two points, a circle, or a sphere.
                double measure = dimension switch
                {
                    1 => 2.0,
                    2 => 2.0 * Math.PI * radius,
                    _ => 4.0 * Math.PI * radius * radius
                };

                integral += value * measure * stepSize;
            }

            Assert.That(integral, Is.EqualTo(1.0).Within(1e-4));
        }

        [Test]
        public void GradientVanishesAtCoincidentPositions()
        {
            // Not just a guard against dividing by zero: df/dq is genuinely zero at the
            // centre, which is the clustering weakness Mueller et al. 2003 worked around.
            Assert.That(math.lengthsq(SphKernel.Gradient(float3.zero, SupportRadius, Normalization3D)), Is.Zero);
            Assert.That(math.lengthsq(SphKernel.Gradient(float2.zero, SupportRadius, Normalization2D)), Is.Zero);
        }

        [Test]
        public void GradientVanishesAtAndBeyondSupportRadius()
        {
            float3 atBoundary = new float3(SupportRadius, 0f, 0f);
            float3 outside = new float3(SupportRadius * 1.5f, 0f, 0f);

            Assert.That(math.lengthsq(SphKernel.Gradient(atBoundary, SupportRadius, Normalization3D)), Is.Zero);
            Assert.That(math.lengthsq(SphKernel.Gradient(outside, SupportRadius, Normalization3D)), Is.Zero);
        }

        [Test]
        public void GradientIsAntisymmetric()
        {
            // This is what makes symmetric SPH forces conserve linear momentum.
            float3 offset = new float3(0.03f, -0.02f, 0.01f);
            float3 forward = SphKernel.Gradient(offset, SupportRadius, Normalization3D);
            float3 reversed = SphKernel.Gradient(-offset, SupportRadius, Normalization3D);

            Assert.That(math.length(forward + reversed), Is.LessThan(1e-6f));
        }

        [Test]
        public void GradientPointsBackTowardsTheNeighbour()
        {
            // offset is x_i - x_j and the kernel decreases with distance, so grad_i W_ij
            // must have a negative component along the offset.
            float3 offset = new float3(0.04f, 0.02f, 0f);
            float3 gradient = SphKernel.Gradient(offset, SupportRadius, Normalization3D);

            Assert.That(math.dot(gradient, offset), Is.LessThan(0f));
        }

        [Test]
        public void GradientMatchesNumericalDerivativeOfKernel()
        {
            const float delta = 1e-4f;
            float normalization = Normalization3D;

            for (float q = 0.05f; q < 0.99f; q += 0.05f)
            {
                float distance = q * SupportRadius;
                float analytic = SphKernel.Gradient(
                    new float3(distance, 0f, 0f), SupportRadius, normalization).x;

                float forward = SphKernel.Evaluate(distance + delta, SupportRadius, normalization);
                float backward = SphKernel.Evaluate(distance - delta, SupportRadius, normalization);
                float numeric = (forward - backward) / (2f * delta);

                float tolerance = math.abs(numeric) * 1e-2f + 1e-2f;
                Assert.That(analytic, Is.EqualTo(numeric).Within(tolerance), $"mismatch at q = {q}");
            }
        }

        [Test]
        public void DensityOnDenseLatticeMatchesRestDensity([Values(2, 3)] int dimension)
        {
            // The single most load-bearing check in this file. If a perfectly sampled
            // lattice does not reconstruct the rest density, no amount of solver tuning
            // will make the simulation settle.
            float spacing = SupportRadius * 0.5f;
            float normalization = SphKernel.Normalization(dimension, SupportRadius);
            float mass = dimension == 2
                ? RestDensity * spacing * spacing
                : RestDensity * spacing * spacing * spacing;

            int range = (int)math.ceil(SupportRadius / spacing);
            int depthRange = dimension == 3 ? range : 0;
            float density = 0f;

            for (int x = -range; x <= range; x++)
            {
                for (int y = -range; y <= range; y++)
                {
                    for (int z = -depthRange; z <= depthRange; z++)
                    {
                        float3 offset = new float3(x, y, z) * spacing;
                        density += mass * SphKernel.Evaluate(
                            math.length(offset), SupportRadius, normalization);
                    }
                }
            }

            Assert.That(density, Is.EqualTo(RestDensity).Within(RestDensity * 0.01f));
        }

        [Test]
        public void LinearFieldIsReproducedOnDenseLattice()
        {
            // With a symmetric kernel and a symmetric sampling both consistency conditions
            // of Eq. (10) hold exactly, so a linear field should come back untouched.
            float error = InterpolationError(SupportRadius, p => 0.5f * (p.x + p.y));

            Assert.That(math.abs(error), Is.LessThan(1e-5f));
        }

        [Test]
        public void QuadraticFieldErrorConvergesAtSecondOrder()
        {
            // A quadratic field cannot be reproduced exactly; the residual is the kernel's
            // smoothing error and must fall off with h^2. Asserting the ratio rather than an
            // absolute bound keeps this independent of the chosen scale.
            Func<float2, float> field = p => 0.5f * (p.x * p.x + p.y * p.y);

            float coarse = math.abs(InterpolationError(SupportRadius, field));
            float fine = math.abs(InterpolationError(SupportRadius * 0.5f, field));

            Assert.That(coarse, Is.GreaterThan(0f), "expected a measurable smoothing error");
            Assert.That(coarse / fine, Is.EqualTo(4f).Within(0.5f));
        }

        [Test]
        public void ComputeKernelMatchesManagedKernel()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Assert.Ignore("Compute shaders are unavailable on this device.");
            }

            var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(ProbeShaderPath);
            Assert.That(shader, Is.Not.Null, $"failed to load {ProbeShaderPath}");

            float3[] offsets = BuildProbeOffsets();
            var offsetBuffer = new ComputeBuffer(offsets.Length, sizeof(float) * 3);
            var resultBuffer = new ComputeBuffer(offsets.Length, sizeof(float) * 4);

            try
            {
                offsetBuffer.SetData(offsets);

                int kernel = shader.FindKernel("EvaluateKernel");
                shader.SetBuffer(kernel, "_SampleOffsets", offsetBuffer);
                shader.SetBuffer(kernel, "_Results", resultBuffer);
                shader.SetFloat("_SupportRadius", SupportRadius);
                shader.SetFloat("_Normalization", Normalization3D);
                shader.SetInt("_SampleCount", offsets.Length);
                shader.Dispatch(kernel, (offsets.Length + 63) / 64, 1, 1);

                var results = new float4[offsets.Length];
                resultBuffer.GetData(results);

                for (int i = 0; i < offsets.Length; i++)
                {
                    float expectedValue = SphKernel.Evaluate(
                        math.length(offsets[i]), SupportRadius, Normalization3D);
                    float3 expectedGradient = SphKernel.Gradient(
                        offsets[i], SupportRadius, Normalization3D);

                    Assert.That(results[i].w, Is.EqualTo(expectedValue).Within(Tolerance(expectedValue)),
                        $"W mismatch at offset {offsets[i]}");

                    for (int axis = 0; axis < 3; axis++)
                    {
                        Assert.That(results[i][axis], Is.EqualTo(expectedGradient[axis])
                                .Within(Tolerance(expectedGradient[axis])),
                            $"grad W component {axis} mismatch at offset {offsets[i]}");
                    }
                }
            }
            finally
            {
                offsetBuffer.Release();
                resultBuffer.Release();
            }
        }

        static float Tolerance(float expected)
        {
            return math.abs(expected) * 1e-3f + 1e-3f;
        }

        static float3[] BuildProbeOffsets()
        {
            var offsets = new float3[128];
            int index = 0;

            // Degenerate and boundary cases first, then a deterministic spread that also
            // reaches outside the support so both early-out branches get exercised.
            offsets[index++] = float3.zero;
            offsets[index++] = new float3(SupportRadius, 0f, 0f);
            offsets[index++] = new float3(SupportRadius * 0.5f, 0f, 0f);
            offsets[index++] = new float3(0f, -SupportRadius * 1.4f, 0f);

            // Fully qualified: Random is ambiguous between System, UnityEngine and Unity.Mathematics.
            var random = new Unity.Mathematics.Random(0x5F3759DFu);
            while (index < offsets.Length)
            {
                offsets[index++] = random.NextFloat3Direction() * random.NextFloat(0f, SupportRadius * 1.3f);
            }

            return offsets;
        }

        /// <summary>
        /// Discretizes <paramref name="field"/> on a dense square lattice and returns the
        /// signed difference between the SPH interpolation and the analytic value at an
        /// interior sample point. Mirrors the experiment in the tutorial's Figs. 3 and 4.
        /// </summary>
        static float InterpolationError(float supportRadius, Func<float2, float> field)
        {
            const int range = 6;

            float spacing = supportRadius * 0.5f;
            float normalization = SphKernel.Normalization(2, supportRadius);
            float mass = RestDensity * spacing * spacing;

            int side = 2 * range + 1;
            var positions = new float2[side * side];
            int index = 0;
            for (int x = -range; x <= range; x++)
            {
                for (int y = -range; y <= range; y++)
                {
                    positions[index++] = new float2(x, y) * spacing;
                }
            }

            var densities = new float[positions.Length];
            for (int i = 0; i < positions.Length; i++)
            {
                float density = 0f;
                for (int j = 0; j < positions.Length; j++)
                {
                    density += mass * SphKernel.Evaluate(
                        math.length(positions[i] - positions[j]), supportRadius, normalization);
                }

                densities[i] = density;
            }

            // Far enough from the lattice edge that every contributing particle, and every
            // density those particles depend on, has a full neighbourhood.
            float2 sample = new float2(2f, 1f) * spacing;
            float interpolated = 0f;
            for (int j = 0; j < positions.Length; j++)
            {
                float weight = SphKernel.Evaluate(
                    math.length(sample - positions[j]), supportRadius, normalization);
                if (weight <= 0f)
                {
                    continue;
                }

                interpolated += field(positions[j]) * (mass / densities[j]) * weight;
            }

            return interpolated - field(sample);
        }
    }
}
