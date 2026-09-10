#ifndef FLUIDSIM_SPH_KERNEL_INCLUDED
#define FLUIDSIM_SPH_KERNEL_INCLUDED

// Cubic spline SPH kernel, Koschier et al. 2019 Eq. (4).
//
// This parametrization makes the smoothing length equal to the support radius,
// so every distance below is compared against h directly and never against h/2.
//
// The dimension only enters through the normalization factor sigma_d, which the
// caller supplies from FluidParameters.KernelNormalization. Keeping it out of
// here means the constant lives in exactly one place.
//
// Mirrored by Runtime/Core/SphKernel.cs. SphKernelTests dispatches SphKernelProbe.compute
// and compares it against the managed version, so the two cannot silently drift apart.

// Radial profile f(q) with q = ||r|| / h. Multiply by sigma_d to get W.
float SphCubicProfile(float q)
{
    if (q >= 1.0)
    {
        return 0.0;
    }

    if (q <= 0.5)
    {
        float q2 = q * q;
        return 6.0 * (q2 * q - q2) + 1.0;
    }

    float t = 1.0 - q;
    return 2.0 * t * t * t;
}

// df/dq. Vanishes at both ends of the support: at q = 1 so the kernel dies off
// smoothly, and at q = 0, which is exactly why pressure forces built on this
// kernel let particles cluster when they get very close (Mueller et al. 2003
// worked around it with a separate spiky kernel).
float SphCubicProfileDerivative(float q)
{
    if (q >= 1.0)
    {
        return 0.0;
    }

    if (q <= 0.5)
    {
        return 6.0 * q * (3.0 * q - 2.0);
    }

    float t = 1.0 - q;
    return -6.0 * t * t;
}

float SphKernel(float dist, float supportRadius, float normalization)
{
    return normalization * SphCubicProfile(dist / supportRadius);
}

// dW/d||r||, negative across the whole support.
float SphKernelDerivative(float dist, float supportRadius, float normalization)
{
    return normalization * SphCubicProfileDerivative(dist / supportRadius) / supportRadius;
}

// grad_i W_ij for offset = x_i - x_j. Points back towards x_j because W decreases
// with distance. Returns zero at coincident positions, where the direction is undefined.
float2 SphKernelGradient(float2 offset, float supportRadius, float normalization)
{
    float distSq = dot(offset, offset);
    if (distSq < 1e-24 || distSq >= supportRadius * supportRadius)
    {
        return float2(0.0, 0.0);
    }

    float dist = sqrt(distSq);
    return (SphKernelDerivative(dist, supportRadius, normalization) / dist) * offset;
}

float3 SphKernelGradient(float3 offset, float supportRadius, float normalization)
{
    float distSq = dot(offset, offset);
    if (distSq < 1e-24 || distSq >= supportRadius * supportRadius)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float dist = sqrt(distSq);
    return (SphKernelDerivative(dist, supportRadius, normalization) / dist) * offset;
}

#endif // FLUIDSIM_SPH_KERNEL_INCLUDED
