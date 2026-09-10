using FluidSim.Core;
using FluidSim.Solver;
using UnityEngine;
using UnityEngine.Rendering;

namespace FluidSim.Debugging
{
    public enum ParticleDebugScalar
    {
        NeighborCount,
        Density,
        Pressure,
        Speed
    }

    /// <summary>
    /// Draws the particles directly from their GPU buffers, coloured by a per-particle scalar.
    ///
    /// Nothing is read back to the CPU: the vertex shader indexes the position buffer by
    /// vertex id and expands a disc around each particle. Colouring by neighbour count makes a
    /// broken neighbour grid visible immediately, which matters a great deal when the solver
    /// itself cannot be stepped through in a debugger.
    /// </summary>
    [AddComponentMenu("FluidSim/Particle Debug Renderer")]
    public sealed class ParticleDebugRenderer : MonoBehaviour
    {
        const string ShaderName = "Hidden/FluidSim/ParticleDebug";
        const int VerticesPerParticle = 6;

        // The position buffer's stride differs between 2D and 3D, so the shader needs to know
        // which layout it is reading.
        const string TwoDimensionalKeyword = "FLUIDSIM_POSITIONS_2D";
        const string ThreeDimensionalKeyword = "FLUIDSIM_POSITIONS_3D";

        static readonly int PositionsId = Shader.PropertyToID("_Positions");
        static readonly int ValuesId = Shader.PropertyToID("_Values");
        static readonly int ParticleRadiusId = Shader.PropertyToID("_ParticleRadius");
        static readonly int PlaneDepthId = Shader.PropertyToID("_PlaneDepth");
        static readonly int ValueRangeId = Shader.PropertyToID("_ValueRange");

        [SerializeField]
        FluidSimulation simulation;

        [SerializeField]
        Shader debugShader;

        [SerializeField]
        ParticleDebugScalar colorBy = ParticleDebugScalar.Density;

        [Tooltip("Derive the colour range from the selected scalar instead of the " +
                 "explicit range below.")]
        [SerializeField]
        bool autoRange = true;

        [SerializeField]
        Vector2 valueRange = new Vector2(0f, 16f);

        [Tooltip("Drawn radius as a multiple of the physical particle radius. Values below one " +
                 "make individual particles easier to tell apart in a dense packing.")]
        [SerializeField, Range(0.2f, 2f)]
        float radiusScale = 1f;

        [Tooltip("Leave this off while the screen-space fluid is enabled. The Game view " +
                 "is the water; these discs stay in the Scene view.")]
        [SerializeField]
        bool drawInGameView;

        Material material;
        LocalKeyword twoDimensionalKeyword;
        LocalKeyword threeDimensionalKeyword;

        void OnEnable()
        {
            if (simulation == null)
            {
                simulation = GetComponent<FluidSimulation>();
            }

            Shader shader = debugShader != null ? debugShader : Shader.Find(ShaderName);
            if (shader == null)
            {
                Debug.LogError($"Could not find the shader '{ShaderName}'. Assign it explicitly.", this);
                enabled = false;
                return;
            }

            material = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
            twoDimensionalKeyword = new LocalKeyword(shader, TwoDimensionalKeyword);
            threeDimensionalKeyword = new LocalKeyword(shader, ThreeDimensionalKeyword);
        }

        void OnDisable()
        {
            if (material != null)
            {
                DestroyImmediate(material);
                material = null;
            }
        }

        void OnRenderObject()
        {
            if (material == null || simulation == null || !simulation.IsReady)
            {
                return;
            }

            // OnRenderObject fires once per rendering camera, including the scene view, which
            // is what makes the particles visible without entering play mode.
            if (Camera.current == null)
            {
                return;
            }

            if (Camera.current.cameraType == CameraType.Game && !ShouldDrawInGameView())
            {
                return;
            }

            var parameters = simulation.Parameters;
            Vector2 range = autoRange ? AutoRange(parameters) : valueRange;

            bool isThreeDimensional = parameters.Dimension == SimulationDimension.Three;
            material.SetKeyword(threeDimensionalKeyword, isThreeDimensional);
            material.SetKeyword(twoDimensionalKeyword, !isThreeDimensional);

            material.SetBuffer(PositionsId, simulation.Particles.SortedPositions);
            material.SetBuffer(ValuesId, ScalarBuffer());

            material.SetFloat(ParticleRadiusId, parameters.ParticleRadius * radiusScale);
            material.SetFloat(PlaneDepthId, parameters.Domain.center.z);
            material.SetVector(ValueRangeId, new Vector4(range.x, range.y, 0f, 0f));

            material.SetPass(0);
            Graphics.DrawProceduralNow(
                MeshTopology.Triangles, simulation.Particles.Count * VerticesPerParticle);

            if (simulation.RigidBody != null && simulation.RigidBody.Count > 0 &&
                simulation.RigidDebugValues != null)
            {
                material.SetBuffer(PositionsId, simulation.RigidBody.Particles.SortedPositions);
                material.SetBuffer(ValuesId, simulation.RigidDebugValues);
                material.SetFloat(ParticleRadiusId, parameters.ParticleRadius * radiusScale * 0.85f);
                material.SetVector(ValueRangeId, new Vector4(0f, 1f, 0f, 0f));
                material.SetPass(0);
                Graphics.DrawProceduralNow(
                    MeshTopology.Triangles, simulation.RigidBody.Count * VerticesPerParticle);
            }
        }

        bool ShouldDrawInGameView()
        {
            if (!drawInGameView)
            {
                return false;
            }

            var fluid = GetComponent<FluidSim.Rendering.ScreenSpaceFluidRenderer>();
            return fluid == null || !fluid.isActiveAndEnabled;
        }

        ComputeBuffer ScalarBuffer()
        {
            switch (colorBy)
            {
                case ParticleDebugScalar.Density:
                    return simulation.Densities;
                case ParticleDebugScalar.Pressure:
                    return simulation.Pressures;
                case ParticleDebugScalar.Speed:
                    return simulation.Speeds;
                default:
                    return simulation.Grid.NeighborCounts;
            }
        }

        Vector2 AutoRange(FluidParameters parameters)
        {
            switch (colorBy)
            {
                case ParticleDebugScalar.Density:
                    // Rest density sits in the teal band. Yellow/red is real
                    // over-density (~15 % and up), not "the fluid has filled in".
                    return new Vector2(parameters.RestDensity * 0.85f, parameters.RestDensity * 1.3f);
                case ParticleDebugScalar.Pressure:
                    return new Vector2(0f, parameters.Stiffness * 0.25f);
                case ParticleDebugScalar.Speed:
                    return new Vector2(0f, 6f);
                default:
                    return new Vector2(0f, Mathf.Max(1f, parameters.ExpectedNeighborCount * 1.5f));
            }
        }
    }
}
