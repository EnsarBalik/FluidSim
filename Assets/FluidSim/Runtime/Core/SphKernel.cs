using System;
using Unity.Mathematics;

namespace FluidSim.Core
{
    /// <summary>
    /// Cubic spline SPH kernel, Koschier et al. 2019 Eq. (4).
    ///
    /// This parametrization makes the smoothing length equal to the support radius,
    /// so every distance is compared against h directly and never against h/2.
    ///
    /// Mirrors Shaders/Include/SphKernel.hlsl. SphKernelTests dispatches the HLSL
    /// version and compares it against this one so the two cannot silently drift apart.
    /// </summary>
    public static class SphKernel
    {
        /// <summary>Radial profile f(q) with q = ||r|| / h. Multiply by sigma_d to get W.</summary>
        public static float Profile(float q)
        {
            if (q >= 1f)
            {
                return 0f;
            }

            if (q <= 0.5f)
            {
                float qSq = q * q;
                return 6f * (qSq * q - qSq) + 1f;
            }

            float t = 1f - q;
            return 2f * t * t * t;
        }

        /// <summary>
        /// df/dq. Vanishes at both ends of the support: at q = 1 so the kernel dies off
        /// smoothly, and at q = 0, which is exactly why pressure forces built on this
        /// kernel let particles cluster when they get very close (Mueller et al. 2003
        /// worked around it with a separate spiky kernel).
        /// </summary>
        public static float ProfileDerivative(float q)
        {
            if (q >= 1f)
            {
                return 0f;
            }

            if (q <= 0.5f)
            {
                return 6f * q * (3f * q - 2f);
            }

            float t = 1f - q;
            return -6f * t * t;
        }

        /// <summary>Normalization factor sigma_d that makes the kernel integrate to one.</summary>
        public static float Normalization(int dimension, float supportRadius)
        {
            switch (dimension)
            {
                case 1:
                    return 4f / (3f * supportRadius);
                case 2:
                    return 40f / (7f * math.PI * supportRadius * supportRadius);
                case 3:
                    return 8f / (math.PI * supportRadius * supportRadius * supportRadius);
                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(dimension), dimension, "Only 1, 2 and 3 dimensions are normalized.");
            }
        }

        public static float Evaluate(float distance, float supportRadius, float normalization)
        {
            return normalization * Profile(distance / supportRadius);
        }

        /// <summary>dW/d||r||, negative across the whole support.</summary>
        public static float Derivative(float distance, float supportRadius, float normalization)
        {
            return normalization * ProfileDerivative(distance / supportRadius) / supportRadius;
        }

        /// <summary>
        /// grad_i W_ij for offset = x_i - x_j. Points back towards x_j because W decreases
        /// with distance. Returns zero at coincident positions, where the direction is undefined.
        /// </summary>
        public static float2 Gradient(float2 offset, float supportRadius, float normalization)
        {
            float distanceSq = math.lengthsq(offset);
            if (distanceSq < 1e-24f || distanceSq >= supportRadius * supportRadius)
            {
                return float2.zero;
            }

            float distance = math.sqrt(distanceSq);
            return Derivative(distance, supportRadius, normalization) / distance * offset;
        }

        /// <inheritdoc cref="Gradient(float2,float,float)"/>
        public static float3 Gradient(float3 offset, float supportRadius, float normalization)
        {
            float distanceSq = math.lengthsq(offset);
            if (distanceSq < 1e-24f || distanceSq >= supportRadius * supportRadius)
            {
                return float3.zero;
            }

            float distance = math.sqrt(distanceSq);
            return Derivative(distance, supportRadius, normalization) / distance * offset;
        }
    }
}
