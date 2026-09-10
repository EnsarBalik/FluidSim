using FluidSim.Core;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Editor
{
    [CustomEditor(typeof(FluidParameters))]
    public sealed class FluidParametersEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var parameters = (FluidParameters)target;
            var resolution = parameters.GridResolution;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Derived", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.FloatField("Particle Size (h~)", parameters.ParticleSize);
                EditorGUILayout.FloatField("Support Radius (h)", parameters.SupportRadius);
                EditorGUILayout.FloatField("Particle Mass", parameters.ParticleMass);
                EditorGUILayout.FloatField("Kernel Normalization", parameters.KernelNormalization);
                EditorGUILayout.FloatField("Expected Neighbors", parameters.ExpectedNeighborCount);
                EditorGUILayout.Vector3IntField("Grid Resolution",
                    new Vector3Int(resolution.x, resolution.y, resolution.z));
                EditorGUILayout.IntField("Cell Count", parameters.CellCount);
                EditorGUILayout.FloatField("Time Step @ 1 m/s", parameters.TimeStepFor(1f));
                EditorGUILayout.FloatField("Time Step @ 10 m/s", parameters.TimeStepFor(10f));
            }

            if (parameters.SurfaceTension > 0f || parameters.Adhesion > 0f ||
                parameters.Vorticity > 0f || parameters.ImplicitViscosityIterations > 0)
            {
                EditorGUILayout.HelpBox(
                    "Surface tension, adhesion, micropolar vorticity and implicit " +
                    "viscosity only run in the DFSPH solver. Weakly Compressible " +
                    "(the Phase 3 checkpoint) ignores them.",
                    MessageType.Info);
            }

            if (parameters.ImplicitViscosityIterations > 0 && parameters.KinematicViscosity < 0.1f)
            {
                EditorGUILayout.HelpBox(
                    "Implicit viscosity is on but Kinematic Viscosity is still near water. " +
                    "Raise it to about 1–5 for honey or paint; otherwise the extra Jacobi " +
                    "passes will not change the look.",
                    MessageType.Info);
            }

            if (parameters.Dimension == SimulationDimension.Two)
            {
                EditorGUILayout.HelpBox(
                    "In 2D the density is a mass per unit area, so the particle mass is " +
                    "rho0 * h~^2 rather than rho0 * h~^3. Mixing the two is why a 2D " +
                    "solver ported from 3D literature never settles at the rest density.",
                    MessageType.Info);
            }

            float expected = parameters.ExpectedNeighborCount;
            bool sparse = parameters.Dimension == SimulationDimension.Three
                ? expected < 25f
                : expected < 9f;

            if (sparse)
            {
                EditorGUILayout.HelpBox(
                    "The support radius gives an unusually small neighborhood. SPH " +
                    "approximations degrade quickly below roughly 30 neighbors in 3D " +
                    "(13 in 2D); consider raising the support radius factor.",
                    MessageType.Warning);
            }
        }
    }
}
