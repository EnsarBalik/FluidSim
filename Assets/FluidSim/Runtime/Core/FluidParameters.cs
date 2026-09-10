using System;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Core
{
    public enum SimulationDimension
    {
        Two = 2,
        Three = 3
    }

    /// <summary>
    /// Single source of truth for the simulation's physical scale.
    /// Only the authored fields are independent; everything the SPH discretization
    /// needs is derived from them, so C#, compute shaders and tests cannot drift apart.
    ///
    /// Notation follows Koschier et al., "SPH Techniques for the Physics Based
    /// Simulation of Fluids and Solids" (Eurographics 2019).
    /// </summary>
    [CreateAssetMenu(fileName = "FluidParameters", menuName = "FluidSim/Fluid Parameters")]
    public sealed class FluidParameters : ScriptableObject
    {
        [Header("Discretization")]
        [Tooltip("Three dimensions is the target. Two exists as a diagnostic mode: the solver " +
                 "is the same source compiled with a different define, and being able to see " +
                 "every single particle is worth a great deal when something is wrong.")]
        [SerializeField]
        SimulationDimension dimension = SimulationDimension.Three;

        [SerializeField, Min(1e-4f)]
        float particleRadius = 0.025f;

        [Tooltip("Support radius as a multiple of the particle size. The tutorial's " +
                 "heuristic is 2, i.e. four times the particle radius, which yields a " +
                 "full neighborhood of roughly 13 particles in 2D and 34 in 3D.")]
        [SerializeField, Min(1f)]
        float supportRadiusFactor = 2f;

        [Header("Material")]
        [SerializeField, Min(1e-3f)]
        float restDensity = 1000f;

        [Tooltip("Kinematic viscosity in m^2/s. Water is about 1e-6. Honey-scale " +
                 "values (1–10) need implicit iterations or the CFL step explodes.")]
        [SerializeField, Min(0f)]
        float kinematicViscosity = 1e-6f;

        [Tooltip("Weiler 2018 implicit viscosity, solved with Jacobi. Zero keeps the " +
                 "explicit Brookshaw term (water). 5–10 plus a large kinematic viscosity " +
                 "gives honey or paint. Only DFSPH uses this.")]
        [SerializeField, Min(0)]
        int implicitViscosityIterations = 0;

        [Tooltip("Stiffness of the state equation p = k * (rho / rho0 - 1). " +
                 "Governs the tolerated density deviation, not the pressure itself: " +
                 "larger values compress less but demand smaller time steps.")]
        [SerializeField, Min(0f)]
        float stiffness = 5000f;

        [Tooltip("Akinci 2013 surface tension, γ. Cohesion plus a colour-field curvature " +
                 "term. Zero turns it off. Water is about 0.07; 0.1 is a visible default " +
                 "at this scale. Only the DFSPH solver uses this.")]
        [SerializeField, Min(0f)]
        float surfaceTension = 0.1f;

        [Tooltip("Akinci 2013 adhesion to boundary particles. Makes the fluid wet the " +
                 "walls instead of beading off. Only the DFSPH solver uses this.")]
        [SerializeField, Min(0f)]
        float adhesion = 0.08f;

        [Tooltip("Bender 2017 micropolar transfer ν_t. Feeds linear vorticity into the " +
                 "microrotation field and back, so SPH viscosity does not kill every swirl. " +
                 "Zero turns the model off. Only DFSPH uses this.")]
        [SerializeField, Min(0f)]
        float vorticity = 0.05f;

        [Tooltip("XSPH viscosity on the angular-velocity field (ζ). Damps neighbour " +
                 "differences in ω without touching linear viscosity.")]
        [SerializeField, Min(0f)]
        float viscosityOmega = 0.1f;

        [Tooltip("Inverse microinertia. 0.5 is Bender's default; larger values make " +
                 "particles spin up faster.")]
        [SerializeField, Min(0f)]
        float inertiaInverse = 0.5f;

        [Header("Time Stepping")]
        [Tooltip("Lambda in the CFL condition dt <= lambda * particleSize / maxSpeed. " +
                 "Monaghan's heuristic is 0.4.")]
        [SerializeField, Range(0.05f, 1f)]
        float cflFactor = 0.4f;

        [SerializeField, Min(1e-6f)]
        float minTimeStep = 1e-5f;

        [SerializeField, Min(1e-6f)]
        float maxTimeStep = 1f / 300f;

        [Header("World")]
        [SerializeField]
        Vector3 gravity = new Vector3(0f, -9.81f, 0f);

        [Tooltip("Default tank. A Fluid Simulation component can override this per object " +
                 "so Play Mode edits do not dirty the shared asset.")]
        [SerializeField]
        Bounds domain = new Bounds(Vector3.zero, new Vector3(6f, 4f, 4f));

        [NonSerialized]
        Bounds runtimeDomain;

        [NonSerialized]
        bool useRuntimeDomain;

        public SimulationDimension Dimension => dimension;

        public int DimensionCount => (int)dimension;

        public float ParticleRadius => particleRadius;

        public float RestDensity => restDensity;

        public float KinematicViscosity => kinematicViscosity;

        public int ImplicitViscosityIterations => implicitViscosityIterations;

        public float Stiffness => stiffness;

        public float SurfaceTension => surfaceTension;

        public float Adhesion => adhesion;

        public float Vorticity => vorticity;

        public float ViscosityOmega => viscosityOmega;

        public float InertiaInverse => inertiaInverse;

        public float CflFactor => cflFactor;

        /// <summary>
        /// Acoustic speed implied by p = k (ρ/ρ0 - 1). WCSPH time steps have to respect
        /// this even when the particles themselves are still, otherwise a pressure wave
        /// can jump a whole smoothing length in one step.
        /// </summary>
        public float SpeedOfSound => math.sqrt(stiffness / math.max(restDensity, 1e-6f));

        public float3 Gravity => gravity;

        public Bounds AuthoredDomain => domain;

        public Bounds Domain => useRuntimeDomain ? runtimeDomain : domain;

        public void SetRuntimeDomain(Bounds bounds)
        {
            Vector3 size = bounds.size;
            float minSize = ParticleSize * 2f;
            size.x = math.max(size.x, minSize);
            size.y = math.max(size.y, minSize);
            size.z = math.max(size.z, minSize);
            if (dimension == SimulationDimension.Two)
            {
                size.z = ParticleSize;
            }

            runtimeDomain = new Bounds(bounds.center, size);
            useRuntimeDomain = true;
        }

        public void ClearRuntimeDomain()
        {
            useRuntimeDomain = false;
        }

        /// <summary>Particle size h~, i.e. the spacing of a dense non-overlapping sampling.</summary>
        public float ParticleSize => 2f * particleRadius;

        /// <summary>
        /// Smoothing length h. The cubic spline kernel is parametrized so that the
        /// smoothing length and the support radius coincide, so this is also the
        /// distance beyond which the kernel vanishes.
        /// </summary>
        public float SupportRadius => supportRadiusFactor * ParticleSize;

        /// <summary>Grid cells are sized to the support radius so a one-ring query suffices.</summary>
        public float CellSize => SupportRadius;

        /// <summary>
        /// Particle mass from the rest density and the volume a particle occupies.
        /// In 2D this is a mass per unit depth (kg/m^2), which is what keeps the 2D
        /// density estimate consistent with the 2D kernel normalization.
        /// </summary>
        public float ParticleMass
        {
            get
            {
                float size = ParticleSize;
                return dimension == SimulationDimension.Two
                    ? restDensity * size * size
                    : restDensity * size * size * size;
            }
        }

        /// <summary>Normalization factor sigma_d of the cubic spline kernel.</summary>
        public float KernelNormalization => SphKernel.Normalization(DimensionCount, SupportRadius);

        /// <summary>
        /// Continuum estimate of how many particles sit inside the support radius, i.e. the
        /// support volume times the number density. This is the figure the tutorial's
        /// "30 to 40 neighbors in 3D" guidance refers to, and it is accurate on average for
        /// an irregular sampling.
        ///
        /// A perfect lattice is a degenerate case and will not match it: whole shells of
        /// lattice points land exactly on the cut-off, where the kernel is zero anyway. A
        /// square lattice at a support radius of two spacings measures 9 rather than 13.
        /// </summary>
        public float ExpectedNeighborCount
        {
            get
            {
                float ratio = SupportRadius / ParticleSize;
                return dimension == SimulationDimension.Two
                    ? math.PI * ratio * ratio
                    : 4f / 3f * math.PI * ratio * ratio * ratio;
            }
        }

        public int3 GridResolution
        {
            get
            {
                int3 resolution = math.max((int3)math.ceil((float3)domain.size / CellSize), 1);
                if (dimension == SimulationDimension.Two)
                {
                    resolution.z = 1;
                }

                return resolution;
            }
        }

        public int CellCount
        {
            get
            {
                int3 resolution = GridResolution;
                return resolution.x * resolution.y * resolution.z;
            }
        }

        /// <summary>CFL bounded time step, clamped to the authored global bounds.</summary>
        public float TimeStepFor(float maxSpeed)
        {
            float cfl = cflFactor * ParticleSize / math.max(maxSpeed, 1e-6f);
            return math.clamp(cfl, minTimeStep, maxTimeStep);
        }

        void OnValidate()
        {
            maxTimeStep = math.max(maxTimeStep, minTimeStep);
        }
    }
}
