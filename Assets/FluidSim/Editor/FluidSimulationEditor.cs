using FluidSim.Core;
using FluidSim.Debugging;
using FluidSim.Rendering;
using FluidSim.Solver;
using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

namespace FluidSim.Editor
{
    [InitializeOnLoad]
    static class FluidSimulationShaderBinder
    {
        static FluidSimulationShaderBinder()
        {
            EditorApplication.delayCall += BindOpenScenes;
        }

        static void BindOpenScenes()
        {
            FluidSimulation[] simulations = Object.FindObjectsByType<FluidSimulation>(
                FindObjectsInactive.Include, FindObjectsSortMode.None);
            foreach (FluidSimulation simulation in simulations)
            {
                var serialized = new SerializedObject(simulation);
                bool rebound = FluidSimulationEditor.BindMissingDfsphShaders(serialized);
                rebound |= FluidSimulationEditor.ApplyPerformanceDefaults(serialized);
                rebound |= FluidSimulationEditor.StampAuthoredDomain(serialized, simulation.Parameters);
                FluidSimulationEditor.EnsureScreenSpaceFluid(simulation, allowAdd: true);
                FluidSimulationEditor.EnsureDemoVisuals(simulation, allowAdd: true);

                if (rebound && simulation.enabled && simulation.gameObject.activeInHierarchy)
                {
                    simulation.enabled = false;
                    simulation.enabled = true;
                }
            }
        }
    }

    /// <summary>
    /// Fills in DFSPH shader references on objects created before Phase 6, then
    /// re-enables the component so Allocate picks the new solver.
    /// </summary>
    [CustomEditor(typeof(FluidSimulation))]
    public sealed class FluidSimulationEditor : UnityEditor.Editor
    {
        const string Dfsph2DPath = "Assets/FluidSim/Shaders/Compute/Dfsph2D.compute";
        const string Dfsph3DPath = "Assets/FluidSim/Shaders/Compute/Dfsph3D.compute";
        const string Diffuse2DPath = "Assets/FluidSim/Shaders/Compute/Diffuse2D.compute";
        const string Diffuse3DPath = "Assets/FluidSim/Shaders/Compute/Diffuse3D.compute";
        const string ParticleDepthPath = "Assets/FluidSim/Shaders/Render/ParticleDepth.shader";
        const string ParticleThicknessPath = "Assets/FluidSim/Shaders/Render/ParticleThickness.shader";
        const string FluidFilterPath = "Assets/FluidSim/Shaders/Render/FluidDepthFilter.shader";
        const string FluidCompositePath = "Assets/FluidSim/Shaders/Render/FluidComposite.shader";
        const string DiffuseParticlePath = "Assets/FluidSim/Shaders/Render/DiffuseParticle.shader";
        const string FoamDepthCopyPath = "Assets/FluidSim/Shaders/Render/FoamDepthCopy.shader";
        const string ShadowBlurPath = "Assets/FluidSim/Shaders/Render/FluidShadowBlur.shader";
        const string SceneBlurPath = "Assets/FluidSim/Shaders/Render/FluidSceneBlur.shader";
        const string DemoSolidPath = "Assets/FluidSim/Shaders/Render/DemoSolid.shader";
        const string DemoFloorPath = "Assets/FluidSim/Shaders/Render/DemoFloor.shader";
        const string DemoGlassPath = "Assets/FluidSim/Shaders/Render/DemoGlass.shader";

        public override void OnInspectorGUI()
        {
            var simulation = (FluidSimulation)target;
            if (BindMissingDfsphShaders(serializedObject)
                | ApplyPerformanceDefaults(serializedObject)
                | StampAuthoredDomain(serializedObject, simulation.Parameters))
            {
                if (simulation.enabled && simulation.gameObject.activeInHierarchy)
                {
                    simulation.enabled = false;
                    simulation.enabled = true;
                }
            }

            EnsureScreenSpaceFluid(simulation, allowAdd: false);
            EnsureDemoVisuals(simulation, allowAdd: false);

            DrawDefaultInspector();

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "Divergence Free is DFSPH. Weakly Compressible is the Phase 3 SESPH " +
                "checkpoint — the same state-equation look, still with Akinci walls. " +
                "The demo rigid box is two-way coupled only in DFSPH.\n\n" +
                "In Play Mode, click and drag the orange box in the Game view. " +
                "Right-mouse looks, WASD flies, Q/E go up and down, Shift boosts.\n\n" +
                "Domain Center / Size is the tank. Drag the cyan box in the Scene view " +
                "or edit the fields; Play Mode keeps the fluid and rebuilds the walls.\n\n" +
                "Diffuse Material is Ihmsen 2012 white-water (spray / foam / bubbles) " +
                "spawned as extra particles after the solver. It does not feed back " +
                "into the fluid.\n\n" +
                "Fluid Demo Visuals draws the glass tank, floor and rigid box in " +
                "the Game view so the water has something to sit in.",
                MessageType.Info);
        }

        void OnSceneGUI()
        {
            var simulation = (FluidSimulation)target;
            SerializedProperty centerProperty = serializedObject.FindProperty("domainCenter");
            SerializedProperty sizeProperty = serializedObject.FindProperty("domainSize");
            if (centerProperty == null || sizeProperty == null)
            {
                return;
            }

            StampAuthoredDomain(serializedObject, simulation.Parameters);

            var handle = new BoxBoundsHandle
            {
                center = centerProperty.vector3Value,
                size = sizeProperty.vector3Value
            };

            Handles.color = new Color(0.35f, 0.75f, 0.95f, 0.95f);
            EditorGUI.BeginChangeCheck();
            handle.DrawHandle();
            if (!EditorGUI.EndChangeCheck())
            {
                return;
            }

            centerProperty.vector3Value = handle.center;
            sizeProperty.vector3Value = Vector3.Max(handle.size, Vector3.one * 0.1f);
            SerializedProperty authored = serializedObject.FindProperty("hasAuthoredDomain");
            if (authored != null)
            {
                authored.boolValue = true;
            }

            serializedObject.ApplyModifiedProperties();
        }

        internal static void EnsureScreenSpaceFluid(FluidSimulation simulation, bool allowAdd)
        {
            if (simulation == null)
            {
                return;
            }

            var depth = AssetDatabase.LoadAssetAtPath<Shader>(ParticleDepthPath);
            var thickness = AssetDatabase.LoadAssetAtPath<Shader>(ParticleThicknessPath);
            var filter = AssetDatabase.LoadAssetAtPath<Shader>(FluidFilterPath);
            var composite = AssetDatabase.LoadAssetAtPath<Shader>(FluidCompositePath);
            var diffuseParticle = AssetDatabase.LoadAssetAtPath<Shader>(DiffuseParticlePath);
            var foamDepthCopy = AssetDatabase.LoadAssetAtPath<Shader>(FoamDepthCopyPath);
            var shadowBlur = AssetDatabase.LoadAssetAtPath<Shader>(ShadowBlurPath);
            var sceneBlur = AssetDatabase.LoadAssetAtPath<Shader>(SceneBlurPath);
            if (depth == null || thickness == null || filter == null || composite == null)
            {
                return;
            }

            var fluid = simulation.GetComponent<ScreenSpaceFluidRenderer>();
            if (fluid == null)
            {
                if (!allowAdd)
                {
                    return;
                }

                fluid = simulation.gameObject.AddComponent<ScreenSpaceFluidRenderer>();
            }

            var serializedFluid = new SerializedObject(fluid);
            bool changed = serializedFluid.FindProperty("simulation").objectReferenceValue != simulation;
            serializedFluid.FindProperty("simulation").objectReferenceValue = simulation;
            changed |= AssignShader(serializedFluid, "depthShader", depth);
            changed |= AssignShader(serializedFluid, "thicknessShader", thickness);
            changed |= AssignShader(serializedFluid, "filterShader", filter);
            changed |= AssignShader(serializedFluid, "compositeShader", composite);
            if (diffuseParticle != null)
            {
                changed |= AssignShader(serializedFluid, "diffuseShader", diffuseParticle);
            }

            if (foamDepthCopy != null)
            {
                changed |= AssignShader(serializedFluid, "foamDepthCopyShader", foamDepthCopy);
            }

            if (shadowBlur != null)
            {
                changed |= AssignShader(serializedFluid, "shadowBlurShader", shadowBlur);
            }

            if (sceneBlur != null)
            {
                changed |= AssignShader(serializedFluid, "sceneBlurShader", sceneBlur);
            }
            if (changed)
            {
                serializedFluid.ApplyModifiedPropertiesWithoutUndo();
            }

            var debug = simulation.GetComponent<ParticleDebugRenderer>();
            if (debug != null)
            {
                var serializedDebug = new SerializedObject(debug);
                SerializedProperty drawInGame = serializedDebug.FindProperty("drawInGameView");
                if (drawInGame != null && drawInGame.boolValue)
                {
                    drawInGame.boolValue = false;
                    serializedDebug.ApplyModifiedPropertiesWithoutUndo();
                }
            }
        }

        internal static void EnsureDemoVisuals(FluidSimulation simulation, bool allowAdd)
        {
            if (simulation == null)
            {
                return;
            }

            var solid = AssetDatabase.LoadAssetAtPath<Shader>(DemoSolidPath);
            var floor = AssetDatabase.LoadAssetAtPath<Shader>(DemoFloorPath);
            var glass = AssetDatabase.LoadAssetAtPath<Shader>(DemoGlassPath);
            if (solid == null || glass == null)
            {
                return;
            }

            var visuals = simulation.GetComponent<FluidDemoVisuals>();
            if (visuals == null)
            {
                if (!allowAdd)
                {
                    return;
                }

                visuals = simulation.gameObject.AddComponent<FluidDemoVisuals>();
            }

            var serialized = new SerializedObject(visuals);
            bool changed = serialized.FindProperty("simulation").objectReferenceValue != simulation;
            serialized.FindProperty("simulation").objectReferenceValue = simulation;
            changed |= AssignShader(serialized, "solidShader", solid);
            if (floor != null)
            {
                changed |= AssignShader(serialized, "floorShader", floor);
            }

            changed |= AssignShader(serialized, "glassShader", glass);
            if (changed)
            {
                serialized.ApplyModifiedPropertiesWithoutUndo();
            }

            EnsureReflectionProbe(simulation);
        }

        static void EnsureReflectionProbe(FluidSimulation simulation)
        {
            Transform existing = simulation.transform.Find("Reflection Probe");
            if (existing != null)
            {
                return;
            }

            var host = new GameObject("Reflection Probe");
            host.transform.SetParent(simulation.transform, false);
            var probe = host.AddComponent<ReflectionProbe>();
            probe.mode = UnityEngine.Rendering.ReflectionProbeMode.Realtime;
            probe.refreshMode = UnityEngine.Rendering.ReflectionProbeRefreshMode.ViaScripting;
            probe.timeSlicingMode = UnityEngine.Rendering.ReflectionProbeTimeSlicingMode.AllFacesAtOnce;
            if (simulation.Parameters != null)
            {
                Bounds domain = simulation.Parameters.Domain;
                host.transform.position = domain.center;
                probe.size = domain.size;
                probe.center = Vector3.zero;
            }

            probe.RenderProbe();
        }

        internal static bool StampAuthoredDomain(SerializedObject serialized, FluidParameters parameters)
        {
            SerializedProperty authored = serialized.FindProperty("hasAuthoredDomain");
            if (authored == null || authored.boolValue || parameters == null)
            {
                return false;
            }

            Bounds domain = parameters.AuthoredDomain;
            serialized.FindProperty("domainCenter").vector3Value = domain.center;
            serialized.FindProperty("domainSize").vector3Value = domain.size;
            authored.boolValue = true;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            return true;
        }

        internal static bool BindMissingDfsphShaders(SerializedObject serialized)
        {
            bool changed = Assign(serialized, "dfsphShader2D", Dfsph2DPath);
            changed |= Assign(serialized, "dfsphShader3D", Dfsph3DPath);
            changed |= Assign(serialized, "diffuseShader2D", Diffuse2DPath);
            changed |= Assign(serialized, "diffuseShader3D", Diffuse3DPath);
            if (changed)
            {
                serialized.ApplyModifiedPropertiesWithoutUndo();
            }

            return changed;
        }

        /// <summary>
        /// One-shot remap of the old authored defaults. Objects that already ran this
        /// stay as the user left them, including a raised substep cap.
        /// </summary>
        internal static bool ApplyPerformanceDefaults(SerializedObject serialized)
        {
            SerializedProperty version = serialized.FindProperty("performanceTuningVersion");
            if (version == null)
            {
                return false;
            }

            bool changed = false;
            if (version.intValue < 1)
            {
                changed |= RemapInt(serialized, "maxSubsteps", 32, 8);
                changed |= RemapInt(serialized, "densityIterations", 5, 3);
                changed |= RemapInt(serialized, "divergenceIterations", 3, 2);
                version.intValue = 1;
            }

            if (version.intValue < 2)
            {
                SerializedProperty drag = serialized.FindProperty("allowGameViewDrag");
                if (drag != null)
                {
                    drag.boolValue = true;
                    changed = true;
                }

                version.intValue = 2;
            }

            if (version.intValue < 3)
            {
                changed |= RemapFloat(serialized, "foamLifetimeMin", 1.5f, 5f);
                changed |= RemapFloat(serialized, "foamLifetimeMax", 4f, 15f);
                changed |= RemapFloat(serialized, "bubbleBuoyancy", 1.2f, 1.5f);
                changed |= RemapFloat(serialized, "bubbleDrag", 0.85f, 3f);
                version.intValue = 3;
            }

            if (changed)
            {
                serialized.ApplyModifiedPropertiesWithoutUndo();
            }

            return changed;
        }

        static bool RemapFloat(SerializedObject serialized, string propertyName, float from, float to)
        {
            SerializedProperty property = serialized.FindProperty(propertyName);
            if (property == null || !Mathf.Approximately(property.floatValue, from))
            {
                return false;
            }

            property.floatValue = to;
            return true;
        }

        static bool RemapInt(SerializedObject serialized, string propertyName, int from, int to)
        {
            SerializedProperty property = serialized.FindProperty(propertyName);
            if (property == null || property.intValue != from)
            {
                return false;
            }

            property.intValue = to;
            return true;
        }

        static bool Assign(SerializedObject serialized, string propertyName, string path)
        {
            SerializedProperty property = serialized.FindProperty(propertyName);
            if (property == null || property.objectReferenceValue != null)
            {
                return false;
            }

            var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
            if (shader == null)
            {
                return false;
            }

            property.objectReferenceValue = shader;
            return true;
        }

        static bool AssignShader(SerializedObject serialized, string propertyName, Shader shader)
        {
            SerializedProperty property = serialized.FindProperty(propertyName);
            if (property == null || property.objectReferenceValue != null)
            {
                return false;
            }

            property.objectReferenceValue = shader;
            return true;
        }
    }
}
