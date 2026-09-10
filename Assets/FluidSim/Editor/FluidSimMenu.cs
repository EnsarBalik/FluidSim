using FluidSim;
using FluidSim.Core;
using FluidSim.Debugging;
using FluidSim.Rendering;
using FluidSim.Solver;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Editor
{
    /// <summary>
    /// Creates a fully wired simulation object. Hooking up the compute shader, the debug shader
    /// and the parameter asset by hand every time is exactly the kind of setup mistake that
    /// looks like a solver bug.
    /// </summary>
    static class FluidSimMenu
    {
        const string NeighborGrid2DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid2D.compute";
        const string NeighborGrid3DPath = "Assets/FluidSim/Shaders/Compute/NeighborGrid3D.compute";
        const string Solver2DPath = "Assets/FluidSim/Shaders/Compute/Sesph2D.compute";
        const string Solver3DPath = "Assets/FluidSim/Shaders/Compute/Sesph3D.compute";
        const string Dfsph2DPath = "Assets/FluidSim/Shaders/Compute/Dfsph2D.compute";
        const string Dfsph3DPath = "Assets/FluidSim/Shaders/Compute/Dfsph3D.compute";
        const string Diffuse2DPath = "Assets/FluidSim/Shaders/Compute/Diffuse2D.compute";
        const string Diffuse3DPath = "Assets/FluidSim/Shaders/Compute/Diffuse3D.compute";
        const string DensityVolume2DPath = "Assets/FluidSim/Shaders/Compute/DensityVolume2D.compute";
        const string DensityVolume3DPath = "Assets/FluidSim/Shaders/Compute/DensityVolume3D.compute";
        const string DebugShaderPath = "Assets/FluidSim/Shaders/Render/ParticleDebug.shader";
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
        const string SettingsDirectory = "Assets/FluidSim/Settings";
        const string ParametersPath = SettingsDirectory + "/FluidParameters.asset";

        [MenuItem("GameObject/FluidSim/Fluid Simulation", false, 10)]
        static void CreateFluidSimulation(MenuCommand command)
        {
            FluidParameters parameters = LoadOrCreateParameters();

            var neighborGrid2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(NeighborGrid2DPath);
            var neighborGrid3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(NeighborGrid3DPath);
            var solver2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Solver2DPath);
            var solver3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Solver3DPath);
            var dfsph2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Dfsph2DPath);
            var dfsph3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Dfsph3DPath);
            var diffuse2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Diffuse2DPath);
            var diffuse3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(Diffuse3DPath);
            var densityVolume2D = AssetDatabase.LoadAssetAtPath<ComputeShader>(DensityVolume2DPath);
            var densityVolume3D = AssetDatabase.LoadAssetAtPath<ComputeShader>(DensityVolume3DPath);
            var debugShader = AssetDatabase.LoadAssetAtPath<Shader>(DebugShaderPath);
            var depthShader = AssetDatabase.LoadAssetAtPath<Shader>(ParticleDepthPath);
            var thicknessShader = AssetDatabase.LoadAssetAtPath<Shader>(ParticleThicknessPath);
            var filterShader = AssetDatabase.LoadAssetAtPath<Shader>(FluidFilterPath);
            var compositeShader = AssetDatabase.LoadAssetAtPath<Shader>(FluidCompositePath);
            var diffuseParticleShader = AssetDatabase.LoadAssetAtPath<Shader>(DiffuseParticlePath);
            var foamDepthCopyShader = AssetDatabase.LoadAssetAtPath<Shader>(FoamDepthCopyPath);
            var shadowBlurShader = AssetDatabase.LoadAssetAtPath<Shader>(ShadowBlurPath);
            var sceneBlurShader = AssetDatabase.LoadAssetAtPath<Shader>(SceneBlurPath);
            var demoSolid = AssetDatabase.LoadAssetAtPath<Shader>(DemoSolidPath);
            var demoFloor = AssetDatabase.LoadAssetAtPath<Shader>(DemoFloorPath);
            var demoGlass = AssetDatabase.LoadAssetAtPath<Shader>(DemoGlassPath);

            if (neighborGrid2D == null || neighborGrid3D == null || solver2D == null || solver3D == null ||
                dfsph2D == null || dfsph3D == null)
            {
                Debug.LogError("Could not load the FluidSim compute shaders. Check " +
                               "Assets/FluidSim/Shaders/Compute.");
                return;
            }

            var host = new GameObject("Fluid Simulation");

            // Wire everything up while the object is inactive, otherwise the components enable
            // themselves against unassigned references and log errors on creation.
            host.SetActive(false);

            var simulation = host.AddComponent<FluidSimulation>();
            var debugRenderer = host.AddComponent<ParticleDebugRenderer>();

            var serializedSimulation = new SerializedObject(simulation);
            serializedSimulation.FindProperty("parameters").objectReferenceValue = parameters;
            serializedSimulation.FindProperty("neighborGridShader2D").objectReferenceValue = neighborGrid2D;
            serializedSimulation.FindProperty("neighborGridShader3D").objectReferenceValue = neighborGrid3D;
            serializedSimulation.FindProperty("solverShader2D").objectReferenceValue = solver2D;
            serializedSimulation.FindProperty("solverShader3D").objectReferenceValue = solver3D;
            serializedSimulation.FindProperty("dfsphShader2D").objectReferenceValue = dfsph2D;
            serializedSimulation.FindProperty("dfsphShader3D").objectReferenceValue = dfsph3D;
            serializedSimulation.FindProperty("diffuseShader2D").objectReferenceValue = diffuse2D;
            serializedSimulation.FindProperty("diffuseShader3D").objectReferenceValue = diffuse3D;
            serializedSimulation.FindProperty("pressureModel").enumValueIndex =
                (int)FluidPressureModel.DivergenceFree;
            serializedSimulation.FindProperty("domainCenter").vector3Value = parameters.AuthoredDomain.center;
            serializedSimulation.FindProperty("domainSize").vector3Value = parameters.AuthoredDomain.size;
            serializedSimulation.FindProperty("hasAuthoredDomain").boolValue = true;
            serializedSimulation.ApplyModifiedPropertiesWithoutUndo();

            var serializedRenderer = new SerializedObject(debugRenderer);
            serializedRenderer.FindProperty("simulation").objectReferenceValue = simulation;
            serializedRenderer.FindProperty("debugShader").objectReferenceValue = debugShader;
            serializedRenderer.FindProperty("drawInGameView").boolValue =
                depthShader == null || thicknessShader == null || filterShader == null || compositeShader == null;
            serializedRenderer.ApplyModifiedPropertiesWithoutUndo();

            if (depthShader != null && thicknessShader != null && filterShader != null && compositeShader != null)
            {
                var fluidRenderer = host.AddComponent<ScreenSpaceFluidRenderer>();
                var serializedFluid = new SerializedObject(fluidRenderer);
                serializedFluid.FindProperty("simulation").objectReferenceValue = simulation;
                serializedFluid.FindProperty("depthShader").objectReferenceValue = depthShader;
                serializedFluid.FindProperty("thicknessShader").objectReferenceValue = thicknessShader;
                serializedFluid.FindProperty("filterShader").objectReferenceValue = filterShader;
                serializedFluid.FindProperty("compositeShader").objectReferenceValue = compositeShader;
                serializedFluid.FindProperty("diffuseShader").objectReferenceValue = diffuseParticleShader;
                serializedFluid.FindProperty("foamDepthCopyShader").objectReferenceValue = foamDepthCopyShader;
                serializedFluid.FindProperty("shadowBlurShader").objectReferenceValue = shadowBlurShader;
                serializedFluid.FindProperty("sceneBlurShader").objectReferenceValue = sceneBlurShader;
                serializedFluid.FindProperty("densityVolumeShader2D").objectReferenceValue = densityVolume2D;
                serializedFluid.FindProperty("densityVolumeShader3D").objectReferenceValue = densityVolume3D;
                serializedFluid.ApplyModifiedPropertiesWithoutUndo();
            }
            else
            {
                Debug.LogWarning(
                    "Screen-space fluid shaders are missing; the Game view will keep the " +
                    "particle debug discs. Check Assets/FluidSim/Shaders/Render.", host);
            }

            if (demoSolid != null && demoGlass != null)
            {
                var visuals = host.AddComponent<FluidDemoVisuals>();
                var serializedVisuals = new SerializedObject(visuals);
                serializedVisuals.FindProperty("simulation").objectReferenceValue = simulation;
                serializedVisuals.FindProperty("solidShader").objectReferenceValue = demoSolid;
                serializedVisuals.FindProperty("floorShader").objectReferenceValue = demoFloor;
                serializedVisuals.FindProperty("glassShader").objectReferenceValue = demoGlass;
                serializedVisuals.ApplyModifiedPropertiesWithoutUndo();
            }

            host.SetActive(true);
            FrameMainCamera(parameters);
            EnsureReflectionProbe(host, parameters);

            GameObjectUtility.SetParentAndAlign(host, command.context as GameObject);
            Undo.RegisterCreatedObjectUndo(host, "Create Fluid Simulation");
            Selection.activeGameObject = host;

            Bounds domain = parameters.Domain;
            string framing = parameters.Dimension == SimulationDimension.Three
                ? $"Frame a perspective camera on {domain.center} from roughly " +
                  $"{domain.size.magnitude:F1} metres away."
                : $"Point an orthographic camera down -Z at {domain.center} with an " +
                  $"orthographic size of at least {domain.extents.y}.";

            Debug.Log(
                $"Created a {parameters.DimensionCount}D fluid simulation over " +
                $"{domain.min} to {domain.max}. {framing}", host);
        }

        static FluidParameters LoadOrCreateParameters()
        {
            // Search the whole project rather than a fixed path, so an asset the user already
            // created somewhere else gets reused instead of quietly duplicated.
            foreach (string guid in AssetDatabase.FindAssets($"t:{nameof(FluidParameters)}"))
            {
                var existing = AssetDatabase.LoadAssetAtPath<FluidParameters>(
                    AssetDatabase.GUIDToAssetPath(guid));
                if (existing != null)
                {
                    return existing;
                }
            }

            if (!AssetDatabase.IsValidFolder(SettingsDirectory))
            {
                AssetDatabase.CreateFolder("Assets/FluidSim", "Settings");
            }

            var created = ScriptableObject.CreateInstance<FluidParameters>();
            AssetDatabase.CreateAsset(created, ParametersPath);
            AssetDatabase.SaveAssets();

            Debug.Log($"Created default fluid parameters at {ParametersPath}.", created);
            return created;
        }

        static void FrameMainCamera(FluidParameters parameters)
        {
            Camera camera = Camera.main;
            if (camera == null || parameters == null)
            {
                return;
            }

            Bounds domain = parameters.AuthoredDomain;
            Vector3 size = domain.size;
            camera.transform.position = domain.center + new Vector3(0f, size.y * 0.35f, -Mathf.Max(size.z, size.x) * 1.15f);
            camera.transform.LookAt(domain.center);
            camera.nearClipPlane = 0.05f;
            camera.farClipPlane = Mathf.Max(size.magnitude * 6f, 40f);

            if (camera.GetComponent<FreeFlyCamera>() == null)
            {
                camera.gameObject.AddComponent<FreeFlyCamera>();
            }
        }

        static void EnsureReflectionProbe(GameObject host, FluidParameters parameters)
        {
            if (host.transform.Find("Reflection Probe") != null)
            {
                return;
            }

            var probeObject = new GameObject("Reflection Probe");
            probeObject.transform.SetParent(host.transform, false);
            var probe = probeObject.AddComponent<ReflectionProbe>();
            probe.mode = UnityEngine.Rendering.ReflectionProbeMode.Realtime;
            probe.refreshMode = UnityEngine.Rendering.ReflectionProbeRefreshMode.ViaScripting;
            probe.timeSlicingMode = UnityEngine.Rendering.ReflectionProbeTimeSlicingMode.AllFacesAtOnce;
            Bounds domain = parameters.AuthoredDomain;
            probeObject.transform.position = domain.center;
            probe.size = domain.size;
            probe.center = Vector3.zero;
            probe.RenderProbe();
        }
    }
}
