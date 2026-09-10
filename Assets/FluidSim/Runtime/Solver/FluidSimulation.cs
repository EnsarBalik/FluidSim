using FluidSim.Core;
using FluidSim.Neighbors;
using FluidSim.Rendering;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.InputSystem;

namespace FluidSim.Solver
{
    public enum FluidPressureModel
    {
        DivergenceFree = 0,
        WeaklyCompressible = 1
    }

    /// <summary>
    /// Owns the simulation state and drives the per-frame GPU work.
    ///
    /// Play mode advances either DFSPH (default) or the Phase 3 SESPH checkpoint, both
    /// with Akinci wall particles. DFSPH can also two-way couple an Akinci-sampled
    /// rigid box. Edit mode only rebuilds the rest-state view.
    ///
    /// The dimension comes from the parameter asset. Both compute shaders are the same source
    /// with a different define, so switching between them is a configuration change rather
    /// than a port.
    /// </summary>
    [AddComponentMenu("FluidSim/Fluid Simulation")]
    [ExecuteAlways]
    public sealed class FluidSimulation : MonoBehaviour
    {
        [SerializeField]
        FluidParameters parameters;

        [Header("Domain")]
        [Tooltip("Centre of the axis-aligned tank, in metres. Play Mode edits stay on " +
                 "this object and do not write back to the Fluid Parameters asset.")]
        [SerializeField]
        Vector3 domainCenter;

        [Tooltip("Size of the tank in metres. Changing this rebuilds the Akinci walls " +
                 "and the neighbour grid; existing fluid particles are kept.")]
        [SerializeField]
        Vector3 domainSize = new Vector3(6f, 4f, 4f);

        [SerializeField, HideInInspector]
        bool hasAuthoredDomain;

        [SerializeField]
        ComputeShader neighborGridShader2D;

        [SerializeField]
        ComputeShader neighborGridShader3D;

        [SerializeField]
        ComputeShader solverShader2D;

        [SerializeField]
        ComputeShader solverShader3D;

        [SerializeField]
        ComputeShader dfsphShader2D;

        [SerializeField]
        ComputeShader dfsphShader3D;

        [SerializeField]
        ComputeShader diffuseShader2D;

        [SerializeField]
        ComputeShader diffuseShader3D;

        [Header("Solver")]
        [Tooltip("Divergence-free SPH is the current solver. Weakly compressible is the " +
                 "Phase 3 checkpoint: state-equation pressure, no Jacobi projection.")]
        [SerializeField]
        FluidPressureModel pressureModel = FluidPressureModel.DivergenceFree;

        [SerializeField, Min(1)]
        int densityIterations = DfsphSolver.DefaultDensityIterations;

        [SerializeField, Min(1)]
        int divergenceIterations = DfsphSolver.DefaultDivergenceIterations;

        [Header("Stepping")]
        [Tooltip("Hard cap on substeps per rendered frame. Keep this modest: a long " +
                 "frame otherwise schedules more work and the FPS collapses.")]
        [SerializeField, Min(1)]
        int maxSubsteps = 8;

        [SerializeField, HideInInspector]
        int performanceTuningVersion = 3;

        [Header("Initial Configuration")]
        [Tooltip("Size of the block of fluid spawned at startup, in metres. The depth is " +
                 "ignored in two dimensions.")]
        [SerializeField]
        Vector3 spawnSize = new Vector3(1.5f, 1.5f, 1.5f);

        [SerializeField]
        Vector3 spawnCenter = new Vector3(-1.5f, 0.5f, 0f);

        [Tooltip("Random displacement as a fraction of the particle spacing. A perfect lattice " +
                 "is a degenerate sampling: whole shells of neighbours land exactly on the " +
                 "support radius, so counts jump around even though the kernel is zero there.")]
        [SerializeField, Range(0f, 0.5f)]
        float spawnJitter = 0.05f;

        [SerializeField]
        uint spawnSeed = 1u;

        [Header("Rigid Body")]
        [Tooltip("Drops an Akinci-sampled box into the dam-break. DFSPH only; " +
                 "the Phase 3 SESPH checkpoint ignores it.")]
        [SerializeField]
        bool spawnDemoRigidBox = true;

        [SerializeField]
        Vector3 rigidBoxSize = new Vector3(0.5f, 0.5f, 0.5f);

        [SerializeField]
        Vector3 rigidBoxCenter = new Vector3(1.2f, -1.55f, 0f);

        [Tooltip("Solid density in kg/m³. Below the rest density the box floats.")]
        [SerializeField, Min(1f)]
        float rigidDensity = 400f;

        [SerializeField]
        bool rigidIsKinematic;

        [Tooltip("Play Mode, Game view: click the orange box and drag it. " +
                 "Release to let DFSPH two-way coupling take over again.")]
        [SerializeField]
        bool allowGameViewDrag = true;

        [Header("Diffuse Material")]
        [Tooltip("Ihmsen 2012 spray, foam and bubbles as a post-process. " +
                 "Does not feed back into the fluid. Off keeps the plain water look.")]
        [SerializeField]
        bool enableDiffuse = true;

        [SerializeField, Min(1024)]
        int maxDiffuseParticles = 65536;

        [SerializeField, Min(0f)]
        float trappedAirMin = 2f;

        [SerializeField, Min(0f)]
        float trappedAirMax = 12f;

        [SerializeField, Min(0f)]
        float waveCrestMin = 2f;

        [SerializeField, Min(0f)]
        float waveCrestMax = 8f;

        [Tooltip("0.5 v² thresholds. Resting water stays at zero.")]
        [SerializeField, Min(0f)]
        float energyMin = 4f;

        [SerializeField, Min(0f)]
        float energyMax = 32f;

        [Tooltip("Maximum trapped-air samples per fluid particle per second. " +
                 "High where fluid pulls apart (wake), not where it piles up.")]
        [SerializeField, Min(0f)]
        float trappedAirRate = 70f;

        [SerializeField, Min(0f)]
        float waveCrestRate = 20f;

        [Tooltip("Seconds of remaining life at spawn. Billboard scale stays full until " +
                 "the last 3 seconds, matching SebLague BillboardFoam.")]
        [SerializeField, Min(0.05f)]
        float foamLifetimeMin = 5f;

        [SerializeField, Min(0.05f)]
        float foamLifetimeMax = 15f;

        [SerializeField, Min(0f)]
        float bubbleBuoyancy = 1.5f;

        [Tooltip("How fast a bubble matches the surrounding fluid velocity. " +
                 "SebLague fluidAccelMul; 3 is the source value.")]
        [SerializeField, Min(0f)]
        float bubbleDrag = 3f;

        public FluidParameters Parameters => parameters;

        public bool SpawnDemoRigidBox => spawnDemoRigidBox;

        public Vector3 DemoRigidBoxSize => rigidBoxSize;

        public Vector3 DemoRigidBoxCenter => rigidBoxCenter;

        public ParticleSet Particles { get; private set; }

        public NeighborGrid Grid { get; private set; }

        public SesphSolver Solver { get; private set; }

        public DfsphSolver DivergenceFreeSolver { get; private set; }

        public ParticleSet Boundary { get; private set; }

        public NeighborGrid BoundaryGrid { get; private set; }

        /// <summary>Constant-zero scalars so the debug view can draw the wall samples.</summary>
        public ComputeBuffer BoundaryDebugValues { get; private set; }

        public DiffuseMaterialSystem Diffuse { get; private set; }

        public SphRigidBody RigidBody { get; private set; }

        public bool IsDraggingRigid { get; private set; }

        /// <summary>Constant ones so the debug view draws the rigid samples warm.</summary>
        public ComputeBuffer RigidDebugValues { get; private set; }

        public bool UsesDivergenceFreeSolver => DivergenceFreeSolver != null;

        public ComputeBuffer Densities => DivergenceFreeSolver?.Densities ?? Solver?.Densities;

        public ComputeBuffer Pressures => DivergenceFreeSolver?.Pressures ?? Solver?.Pressures;

        public ComputeBuffer Speeds => DivergenceFreeSolver?.Speeds ?? Solver?.Speeds;

        public bool IsReady =>
            Particles != null && Grid != null && Particles.Count > 0 &&
            (DivergenceFreeSolver != null || Solver != null);

        Vector3 appliedDomainCenter;
        Vector3 appliedDomainSize;
        bool domainApplied;
        Vector3 rigidGrabOffset;
        Plane rigidDragPlane;

        void OnEnable()
        {
            SyncDomain(allocateIfReady: false);
            Allocate();
        }

        void OnDisable()
        {
            parameters?.ClearRuntimeDomain();
            domainApplied = false;
            Release();
        }

        void OnValidate()
        {
            domainSize.x = math.max(domainSize.x, 0.1f);
            domainSize.y = math.max(domainSize.y, 0.1f);
            domainSize.z = math.max(domainSize.z, 0.1f);
        }

        void Update()
        {
            SyncDomain(allocateIfReady: true);

            if (!IsReady)
            {
                return;
            }

            if (!ActiveSolverMatchesModel())
            {
                Allocate();
                if (!IsReady)
                {
                    return;
                }
            }

            if (!Application.isPlaying)
            {
                // Edit mode keeps the rest-state view alive without integrating, so the
                // Scene view stays useful while the spawn region is being authored.
                IsDraggingRigid = false;
                SyncRigidToAuthoredPose();
                Rebuild(countNeighbors: true);
                ComputeActiveDensity();
                return;
            }

            HandleRigidDrag();

            float remaining = Time.deltaTime;
            float dt = parameters.TimeStepFor(EstimatedMaxSpeed());
            int steps = 0;

            while (remaining > 0f && steps < maxSubsteps)
            {
                float step = math.min(dt, remaining);
                Grid.Build(Particles, parameters);
                PrepareRigidForStep();
                AdvanceActiveSolver(step);
                IntegrateRigid(step);
                remaining -= step;
                steps++;
            }

            // DFSPH EndStep does not move particles, so the mid-step grid and
            // density are still valid. SESPH integrated in place and needs a rebuild.
            if (DivergenceFreeSolver == null)
            {
                Grid.Build(Particles, parameters);
                ComputeActiveDensity();
            }

            Grid.CountNeighbors(Particles, parameters);
            TickDiffuse(Time.deltaTime);
        }

        void Rebuild(bool countNeighbors)
        {
            Grid.Build(Particles, parameters);
            if (countNeighbors)
            {
                Grid.CountNeighbors(Particles, parameters);
            }
        }

        bool ActiveSolverMatchesModel()
        {
            ComputeShader dfsphShader = parameters.Dimension == SimulationDimension.Three
                ? dfsphShader3D
                : dfsphShader2D;
            bool wantDfsph = pressureModel == FluidPressureModel.DivergenceFree && dfsphShader != null;
            return wantDfsph ? DivergenceFreeSolver != null : Solver != null;
        }

        void AdvanceActiveSolver(float step)
        {
            if (DivergenceFreeSolver != null)
            {
                DivergenceFreeSolver.BeginStep(
                    Particles, Grid, parameters, step, densityIterations);
                Grid.Build(Particles, parameters);
                DivergenceFreeSolver.EndStep(
                    Particles, Grid, parameters, step, divergenceIterations);
                return;
            }

            Solver.Advance(Particles, Grid, parameters, step);
        }

        void ComputeActiveDensity()
        {
            if (DivergenceFreeSolver != null)
            {
                DivergenceFreeSolver.ComputeDensity(Particles, Grid, parameters);
                return;
            }

            Solver.ComputeDensity(Particles, Grid, parameters);
        }

        float EstimatedMaxSpeed()
        {
            // Free-fall from the top of the domain. Tracking the true max on the GPU
            // would stall a readback; this stays conservative enough without a device sync.
            float fall = math.sqrt(2f * math.length(parameters.Gravity) * parameters.Domain.size.y);

            // WCSPH also has to resolve the EOS sound speed, even at rest. DFSPH has no
            // acoustic wave, so the same splash is allowed a larger step.
            return DivergenceFreeSolver != null
                ? math.max(fall, 8f)
                : math.max(parameters.SpeedOfSound, fall);
        }

        void Allocate()
        {
            Release();

            if (parameters == null)
            {
                Debug.LogError($"{nameof(FluidSimulation)} needs a {nameof(FluidParameters)} asset.", this);
                return;
            }

            if (!SystemInfo.supportsComputeShaders)
            {
                Debug.LogError("This device does not support compute shaders.", this);
                return;
            }

            bool isThreeDimensional = parameters.Dimension == SimulationDimension.Three;
            ComputeShader gridShader = isThreeDimensional ? neighborGridShader3D : neighborGridShader2D;
            ComputeShader sesphShader = isThreeDimensional ? solverShader3D : solverShader2D;
            ComputeShader dfsphShader = isThreeDimensional ? dfsphShader3D : dfsphShader2D;
            bool useDfsph = pressureModel == FluidPressureModel.DivergenceFree && dfsphShader != null;
            ComputeShader solverShader = useDfsph ? dfsphShader : sesphShader;
            if (gridShader == null || solverShader == null)
            {
                Debug.LogError(
                    $"The parameters describe a {parameters.DimensionCount}D simulation but the " +
                    "matching compute shaders are not assigned. Recreate the object from " +
                    "GameObject > FluidSim > Fluid Simulation.", this);
                return;
            }

            if (pressureModel == FluidPressureModel.DivergenceFree && !useDfsph)
            {
                Debug.LogWarning(
                    "DFSPH shaders are not assigned; falling back to the Phase 3 SESPH solver. " +
                    "Recreate the object from GameObject > FluidSim > Fluid Simulation.", this);
            }

            int3 counts = SpawnCounts();
            int total = counts.x * counts.y * counts.z;
            if (total <= 0)
            {
                Debug.LogWarning("The spawn region is smaller than a single particle.", this);
                return;
            }

            float3[] lattice = BuildSpawnLattice(counts);

            Particles = new ParticleSet(total, parameters.DimensionCount);
            if (isThreeDimensional)
            {
                Particles.Fill(lattice);
            }
            else
            {
                var flattened = new float2[total];
                for (int i = 0; i < total; i++)
                {
                    flattened[i] = lattice[i].xy;
                }

                Particles.Fill(flattened);
            }

            Grid = new NeighborGrid(gridShader, parameters, total);
            if (useDfsph)
            {
                DivergenceFreeSolver = new DfsphSolver(solverShader, total, parameters.DimensionCount);
            }
            else
            {
                Solver = new SesphSolver(solverShader, total, parameters.DimensionCount);
            }

            AllocateBoundary(gridShader, isThreeDimensional);
            AllocateRigid(gridShader);
            AllocateDiffuse();

            // Build immediately so the sorted buffers are meaningful before the first Update,
            // which in edit mode may not run until something forces a repaint.
            Rebuild(countNeighbors: true);
            ComputeActiveDensity();
        }

        void AllocateBoundary(ComputeShader gridShader, bool isThreeDimensional)
        {
            float3[] samples = DomainBoundarySampler.Sample(parameters);
            if (samples.Length == 0)
            {
                return;
            }

            Boundary = new ParticleSet(samples.Length, parameters.DimensionCount);
            if (isThreeDimensional)
            {
                Boundary.Fill(samples);
            }
            else
            {
                var flattened = new float2[samples.Length];
                for (int i = 0; i < samples.Length; i++)
                {
                    flattened[i] = samples[i].xy;
                }

                Boundary.Fill(flattened);
            }

            BoundaryGrid = new NeighborGrid(gridShader, parameters, samples.Length);
            BoundaryGrid.Build(Boundary, parameters);

            BoundaryDebugValues = new ComputeBuffer(samples.Length, sizeof(float));
            BoundaryDebugValues.SetData(new float[samples.Length]);

            if (DivergenceFreeSolver != null)
            {
                DivergenceFreeSolver.SetBoundary(Boundary, BoundaryGrid, parameters);
            }
            else
            {
                Solver.SetBoundary(Boundary, BoundaryGrid, parameters);
            }
        }

        void AllocateRigid(ComputeShader gridShader)
        {
            if (!spawnDemoRigidBox || DivergenceFreeSolver == null)
            {
                return;
            }

            float3 size = rigidBoxSize;
            float3 center = rigidBoxCenter;
            if (parameters.Dimension == SimulationDimension.Two)
            {
                size.z = parameters.ParticleSize;
                center.z = parameters.Domain.center.z;
            }

            RigidBody = new SphRigidBody(
                gridShader, parameters, size, center, rigidDensity, rigidIsKinematic);
            RigidDebugValues = new ComputeBuffer(RigidBody.Count, sizeof(float));
            var ones = new float[RigidBody.Count];
            for (int i = 0; i < ones.Length; i++)
            {
                ones[i] = 1f;
            }

            RigidDebugValues.SetData(ones);
            DivergenceFreeSolver.SetRigid(RigidBody, parameters);
        }

        void SyncRigidToAuthoredPose()
        {
            if (RigidBody == null)
            {
                return;
            }

            float3 center = rigidBoxCenter;
            if (parameters.Dimension == SimulationDimension.Two)
            {
                center.z = parameters.Domain.center.z;
            }

            RigidBody.SetPose(center, quaternion.identity);
            PrepareRigidForStep();
        }

        void PrepareRigidForStep()
        {
            if (RigidBody == null || DivergenceFreeSolver == null)
            {
                return;
            }

            bool rebuild = !RigidBody.IsKinematic || !RigidBody.GridReady;
            if (rebuild)
            {
                RigidBody.RebuildGrid(parameters);
                DivergenceFreeSolver.SetRigid(RigidBody, parameters);
            }
            else if (!RigidBody.MassesCaptured)
            {
                DivergenceFreeSolver.SetRigid(RigidBody, parameters);
            }

            DivergenceFreeSolver.ClearRigidForces();
        }

        void IntegrateRigid(float step)
        {
            if (RigidBody == null || DivergenceFreeSolver == null ||
                RigidBody.IsKinematic || IsDraggingRigid)
            {
                return;
            }

            DivergenceFreeSolver.ReadRigidForceAndTorque(RigidBody.Position, out float3 force, out float3 torque);
            RigidBody.Integrate(force, torque, step, parameters.Gravity, parameters.Domain);
        }

        void HandleRigidDrag()
        {
            if (!allowGameViewDrag || RigidBody == null || parameters == null)
            {
                IsDraggingRigid = false;
                return;
            }

            Mouse mouse = Mouse.current;
            Camera camera = Camera.main;
            if (mouse == null || camera == null)
            {
                IsDraggingRigid = false;
                return;
            }

            if (!IsDraggingRigid)
            {
                if (!mouse.leftButton.wasPressedThisFrame ||
                    mouse.rightButton.isPressed ||
                    !PointerIsOverGameView())
                {
                    return;
                }

                Ray ray = camera.ScreenPointToRay(mouse.position.ReadValue());
                if (!RigidBody.TryRaycast(ray, out float hitDistance))
                {
                    return;
                }

                Vector3 hit = ray.GetPoint(hitDistance);
                rigidGrabOffset = hit - (Vector3)RigidBody.Position;
                rigidDragPlane = new Plane(-camera.transform.forward, hit);
                RigidBody.LinearVelocity = float3.zero;
                RigidBody.AngularVelocity = float3.zero;
                IsDraggingRigid = true;
                return;
            }

            if (!mouse.leftButton.isPressed)
            {
                IsDraggingRigid = false;
                return;
            }

            Ray dragRay = camera.ScreenPointToRay(mouse.position.ReadValue());
            if (!rigidDragPlane.Raycast(dragRay, out float planeDistance))
            {
                return;
            }

            Vector3 target = dragRay.GetPoint(planeDistance) - rigidGrabOffset;
            RigidBody.DragTo(target, Time.deltaTime, parameters.Domain);
        }

        static bool PointerIsOverGameView()
        {
#if UNITY_EDITOR
            System.Type windowType = System.Type.GetType("UnityEditor.EditorWindow,UnityEditor");
            object hovered = windowType?
                .GetProperty("mouseOverWindow",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)?
                .GetValue(null, null);
            return hovered != null && hovered.GetType().Name == "GameView";
#else
            return true;
#endif
        }

        void TickDiffuse(float deltaTime)
        {
            if (!enableDiffuse || Particles == null || Grid == null)
            {
                Diffuse?.Dispose();
                Diffuse = null;
                return;
            }

            AllocateDiffuse();
            if (Diffuse == null)
            {
                return;
            }

            Diffuse.Step(Particles, Grid, parameters, new DiffuseMaterialSettings
            {
                TrappedAirMin = trappedAirMin,
                TrappedAirMax = math.max(trappedAirMax, trappedAirMin + 1e-3f),
                WaveCrestMin = waveCrestMin,
                WaveCrestMax = math.max(waveCrestMax, waveCrestMin + 1e-3f),
                EnergyMin = energyMin,
                EnergyMax = math.max(energyMax, energyMin + 1e-3f),
                TrappedAirRate = trappedAirRate,
                WaveCrestRate = waveCrestRate,
                FoamLifeMin = foamLifetimeMin,
                FoamLifeMax = math.max(foamLifetimeMax, foamLifetimeMin),
                BubbleBuoyancy = bubbleBuoyancy,
                BubbleDrag = bubbleDrag
            }, deltaTime);
        }

        void AllocateDiffuse()
        {
            if (!enableDiffuse || Particles == null || parameters == null)
            {
                Diffuse?.Dispose();
                Diffuse = null;
                return;
            }

            ComputeShader shader = parameters.Dimension == SimulationDimension.Three
                ? diffuseShader3D
                : diffuseShader2D;
            if (shader == null)
            {
                return;
            }

            if (Diffuse != null &&
                Diffuse.Capacity == maxDiffuseParticles &&
                Diffuse.FluidCount == Particles.Count)
            {
                return;
            }

            Diffuse?.Dispose();
            Diffuse = new DiffuseMaterialSystem(shader, Particles.Count, maxDiffuseParticles);
        }

        void Release()
        {
            Diffuse?.Dispose();
            Diffuse = null;

            DivergenceFreeSolver?.Dispose();
            DivergenceFreeSolver = null;

            Solver?.Dispose();
            Solver = null;

            IsDraggingRigid = false;
            RigidBody?.Dispose();
            RigidBody = null;

            RigidDebugValues?.Release();
            RigidDebugValues = null;

            BoundaryGrid?.Dispose();
            BoundaryGrid = null;

            Boundary?.Dispose();
            Boundary = null;

            BoundaryDebugValues?.Release();
            BoundaryDebugValues = null;

            Grid?.Dispose();
            Grid = null;

            Particles?.Dispose();
            Particles = null;
        }

        int3 SpawnCounts()
        {
            float spacing = parameters.ParticleSize;
            int3 counts = new int3(
                Mathf.FloorToInt(spawnSize.x / spacing),
                Mathf.FloorToInt(spawnSize.y / spacing),
                parameters.Dimension == SimulationDimension.Three
                    ? Mathf.FloorToInt(spawnSize.z / spacing)
                    : 1);

            return math.max(counts, 0);
        }

        /// <summary>
        /// A dense lattice at exactly the particle spacing, which is the sampling the rest
        /// density is defined against. Coordinates are always three dimensional; the depth is
        /// collapsed to a single layer in two dimensions.
        /// </summary>
        float3[] BuildSpawnLattice(int3 counts)
        {
            float spacing = parameters.ParticleSize;
            var positions = new float3[counts.x * counts.y * counts.z];

            float3 origin = (float3)spawnCenter - 0.5f * (float3)(counts - 1) * spacing;

            // Keep everything strictly inside the domain so the very first grid build cannot
            // depend on the out-of-bounds clamp in SphCellCoord.
            Bounds domain = parameters.Domain;
            float margin = parameters.ParticleRadius;
            float3 lower = (float3)domain.min + margin;
            float3 upper = (float3)domain.max - margin;

            var random = new Unity.Mathematics.Random(spawnSeed == 0u ? 1u : spawnSeed);
            float jitter = spawnJitter * spacing;

            int index = 0;
            for (int z = 0; z < counts.z; z++)
            {
                for (int y = 0; y < counts.y; y++)
                {
                    for (int x = 0; x < counts.x; x++)
                    {
                        float3 offset = jitter > 0f
                            ? random.NextFloat3(new float3(-jitter), new float3(jitter))
                            : float3.zero;

                        if (parameters.Dimension == SimulationDimension.Two)
                        {
                            offset.z = 0f;
                        }

                        float3 position = origin + new float3(x, y, z) * spacing + offset;
                        positions[index++] = math.clamp(position, lower, upper);
                    }
                }
            }

            return positions;
        }

        [ContextMenu("Log Neighbour Statistics")]
        void LogNeighborStatistics()
        {
            if (!IsReady)
            {
                Debug.LogWarning("The simulation is not running.", this);
                return;
            }

            float[] counts = Grid.ReadNeighborCounts(Particles.Count);

            float minimum = float.MaxValue;
            float maximum = float.MinValue;
            double total = 0.0;
            foreach (float count in counts)
            {
                minimum = math.min(minimum, count);
                maximum = math.max(maximum, count);
                total += count;
            }

            Debug.Log(
                $"{parameters.DimensionCount}D, {counts.Length} particles. Neighbours: min " +
                $"{minimum}, mean {total / counts.Length:F2}, max {maximum}. The continuum " +
                $"estimate for an irregular sampling is {parameters.ExpectedNeighborCount:F2}; " +
                "particles near the surface sit well below it because half their neighbourhood " +
                "is empty.", this);
        }

        [ContextMenu("Verify Grid Against Brute Force")]
        void VerifyGridAgainstBruteForce()
        {
            if (!IsReady)
            {
                Debug.LogWarning("The simulation is not running.", this);
                return;
            }

            Grid.Build(Particles, parameters);
            Grid.CountNeighbors(Particles, parameters);
            Grid.CountNeighborsReference(Particles, parameters);

            float[] fromGrid = Grid.ReadNeighborCounts(Particles.Count);
            float[] reference = Grid.ReadReferenceNeighborCounts(Particles.Count);

            int mismatches = 0;
            for (int i = 0; i < fromGrid.Length; i++)
            {
                if (fromGrid[i] != reference[i])
                {
                    mismatches++;
                }
            }

            if (mismatches == 0)
            {
                Debug.Log($"Grid matches brute force on all {fromGrid.Length} particles.", this);
            }
            else
            {
                Debug.LogError(
                    $"Grid disagrees with brute force on {mismatches} of {fromGrid.Length} particles.",
                    this);
            }
        }

        void EnsureAuthoredDomain()
        {
            if (hasAuthoredDomain || parameters == null)
            {
                return;
            }

            Bounds authored = parameters.AuthoredDomain;
            domainCenter = authored.center;
            domainSize = authored.size;
            hasAuthoredDomain = true;
        }

        Bounds AuthoredDomainBounds()
        {
            return new Bounds(domainCenter, domainSize);
        }

        void SyncDomain(bool allocateIfReady)
        {
            if (parameters == null)
            {
                return;
            }

            EnsureAuthoredDomain();
            bool fieldsChanged = !domainApplied ||
                                 appliedDomainCenter != domainCenter ||
                                 appliedDomainSize != domainSize;

            parameters.SetRuntimeDomain(AuthoredDomainBounds());
            if (!fieldsChanged)
            {
                return;
            }

            int previousCells = Grid != null ? Grid.CellCount : -1;
            appliedDomainCenter = domainCenter;
            appliedDomainSize = domainSize;
            domainApplied = true;

            if (!allocateIfReady || !IsReady)
            {
                return;
            }

            RebuildDomainResources(previousCells != parameters.CellCount);
        }

        void RebuildDomainResources(bool reallocateGrids)
        {
            bool isThreeDimensional = parameters.Dimension == SimulationDimension.Three;
            ComputeShader gridShader = isThreeDimensional ? neighborGridShader3D : neighborGridShader2D;
            if (gridShader == null)
            {
                return;
            }

            if (reallocateGrids && Grid != null)
            {
                int capacity = Grid.Capacity;
                Grid.Dispose();
                Grid = new NeighborGrid(gridShader, parameters, capacity);
                RigidBody?.ReallocateGrid(parameters);
            }

            if (RigidBody != null)
            {
                RigidBody.RebuildGrid(parameters);
            }

            BoundaryGrid?.Dispose();
            Boundary?.Dispose();
            BoundaryDebugValues?.Release();
            BoundaryGrid = null;
            Boundary = null;
            BoundaryDebugValues = null;
            AllocateBoundary(gridShader, isThreeDimensional);
        }

        void OnDrawGizmos()
        {
            Vector3 tankCenter = domainCenter;
            Vector3 tankSize = domainSize;
            if (parameters != null && parameters.Dimension == SimulationDimension.Two)
            {
                tankSize.z = 0f;
                tankCenter.z = parameters.Domain.center.z;
            }

            Gizmos.color = new Color(0.35f, 0.75f, 0.95f, 0.9f);
            Gizmos.matrix = Matrix4x4.identity;
            Gizmos.DrawWireCube(tankCenter, tankSize);

            if (!spawnDemoRigidBox)
            {
                return;
            }

            float3 center = Application.isPlaying && RigidBody != null
                ? RigidBody.Position
                : (float3)rigidBoxCenter;
            quaternion rotation = Application.isPlaying && RigidBody != null
                ? RigidBody.Rotation
                : quaternion.identity;
            Vector3 size = rigidBoxSize;
            if (parameters != null && parameters.Dimension == SimulationDimension.Two)
            {
                size.z = 0f;
                center.z = parameters.Domain.center.z;
            }

            Gizmos.color = new Color(0.92f, 0.45f, 0.16f, 0.85f);
            Gizmos.matrix = Matrix4x4.TRS((Vector3)center, (Quaternion)rotation, Vector3.one);
            Gizmos.DrawWireCube(Vector3.zero, size);
        }
    }
}
