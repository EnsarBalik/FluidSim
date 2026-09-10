using System.Collections.Generic;
using FluidSim.Core;
using FluidSim.Solver;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

namespace FluidSim.Rendering
{
    public enum FluidDebugView
    {
        Composite = 0,
        Depth = 1,
        Thickness = 2,
        Normals = 3,
        ThicknessRaw = 4,
        Foam = 5,
        Reflection = 6
    }

    public enum FluidThicknessMode
    {
        OverlapCount = 0,
        SphereChord = 1
    }

    public enum FluidSurfaceSmoothType
    {
        Bilateral = 0,
        Gaussian = 1
    }

    /// <summary>
    /// Screen-space fluid for the Built-in pipeline. Particle spheres still fill
    /// the Green depth/thickness targets for debug views, foam and the sun
    /// shadow. Game-view shading is SebLague RayMarchingTest: SPH density is
    /// voxelised into a 3D map, rays walk the isosurface (density − offset),
    /// and Fresnel picks the more interesting of reflect vs refract so clustered
    /// particles read as one volume rather than separate discs.
    /// </summary>
    [AddComponentMenu("FluidSim/Screen Space Fluid Renderer")]
    [ExecuteAlways]
    public sealed class ScreenSpaceFluidRenderer : MonoBehaviour
    {
        const string DepthShaderName = "Hidden/FluidSim/ParticleDepth";
        const string ThicknessShaderName = "Hidden/FluidSim/ParticleThickness";
        const string FilterShaderName = "Hidden/FluidSim/FluidDepthFilter";
        const string CompositeShaderName = "Hidden/FluidSim/FluidComposite";
        const string DiffuseShaderName = "Hidden/FluidSim/DiffuseParticle";
        const string FoamDepthCopyShaderName = "Hidden/FluidSim/FoamDepthCopy";
        const string ShadowBlurShaderName = "Hidden/FluidSim/FluidShadowBlur";
        const string SceneBlurShaderName = "Hidden/FluidSim/FluidSceneBlur";
        const string ShadowPassKeywordName = "FLUIDSIM_SHADOW_PASS";
        const int VerticesPerParticle = 6;
        const int FilterPackPass = 0;
        const int FilterHorizontalPass = 1;
        const int FilterVerticalPass = 2;
        const CameraEvent InjectionPoint = CameraEvent.BeforeImageEffects;
        const float EmptyDepth = 1e7f;

        const string TwoDimensionalKeyword = "FLUIDSIM_POSITIONS_2D";
        const string ThreeDimensionalKeyword = "FLUIDSIM_POSITIONS_3D";

        static readonly int PositionsId = Shader.PropertyToID("_Positions");
        static readonly int ParticleRadiusId = Shader.PropertyToID("_ParticleRadius");
        static readonly int PlaneDepthId = Shader.PropertyToID("_PlaneDepth");
        static readonly int ThicknessContributionId = Shader.PropertyToID("_ThicknessContribution");
        static readonly int ThicknessModeId = Shader.PropertyToID("_ThicknessMode");
        static readonly int WorldFilterRadiusId = Shader.PropertyToID("_WorldFilterRadius");
        static readonly int MaxScreenSpaceRadiusId = Shader.PropertyToID("_MaxScreenSpaceRadius");
        static readonly int FilterStrengthId = Shader.PropertyToID("_FilterStrength");
        static readonly int FilterDiffStrengthId = Shader.PropertyToID("_FilterDiffStrength");
        static readonly int FilterBilateralId = Shader.PropertyToID("_FilterBilateral");
        static readonly int CameraProjectionM00Id = Shader.PropertyToID("_CameraProjectionM00");
        static readonly int FluidRawDepthId = Shader.PropertyToID("_FluidRawDepth");
        static readonly int FluidRawThicknessId = Shader.PropertyToID("_FluidRawThickness");
        static readonly int FluidSceneColorId = Shader.PropertyToID("_FluidSceneColor");
        static readonly int FluidSceneBlurId = Shader.PropertyToID("_FluidSceneBlur");
        static readonly int FluidFoamId = Shader.PropertyToID("_FluidFoam");
        static readonly int AbsorbColorId = Shader.PropertyToID("_AbsorbColor");
        static readonly int ScatterColorId = Shader.PropertyToID("_ScatterColor");
        static readonly int LightDirectionVsId = Shader.PropertyToID("_LightDirectionVS");
        static readonly int RefractionMultiplierId = Shader.PropertyToID("_RefractionMultiplier");
        static readonly int SpecularPowerId = Shader.PropertyToID("_SpecularPower");
        static readonly int SpecularIntensityId = Shader.PropertyToID("_SpecularIntensity");
        static readonly int FresnelStrengthId = Shader.PropertyToID("_FresnelStrength");
        static readonly int ThicknessScaleId = Shader.PropertyToID("_ThicknessScale");
        static readonly int DensityMultiplierId = Shader.PropertyToID("_DensityMultiplier");
        static readonly int DebugViewId = Shader.PropertyToID("_DebugView");
        static readonly int DomainCenterId = Shader.PropertyToID("_DomainCenter");
        static readonly int DomainSizeId = Shader.PropertyToID("_DomainSize");
        static readonly int FluidCameraVpId = Shader.PropertyToID("_FluidCameraVP");
        static readonly int FluidInvPId = Shader.PropertyToID("_FluidInvP");
        static readonly int FluidCameraToWorldId = Shader.PropertyToID("_FluidCameraToWorld");
        static readonly int FluidWorldToCameraId = Shader.PropertyToID("_FluidWorldToCamera");
        static readonly int RayMarchStepId = Shader.PropertyToID("_RayMarchStep");
        static readonly int RayMarchStepsId = Shader.PropertyToID("_RayMarchSteps");
        static readonly int BackgroundBlurId = Shader.PropertyToID("_BackgroundBlur");
        static readonly int BackgroundRevealId = Shader.PropertyToID("_BackgroundReveal");
        static readonly int InteriorMixId = Shader.PropertyToID("_InteriorMix");
        static readonly int DensityMapId = Shader.PropertyToID("DensityMap");
        static readonly int DensityOffsetId = Shader.PropertyToID("_DensityOffset");
        static readonly int LightStepSizeId = Shader.PropertyToID("_LightStepSize");
        static readonly int IndexOfRefractionId = Shader.PropertyToID("_IndexOfRefraction");
        static readonly int NumRefractionsId = Shader.PropertyToID("_NumRefractions");
        static readonly int DirToSunId = Shader.PropertyToID("_DirToSun");
        static readonly int CubeLocalToWorldId = Shader.PropertyToID("_CubeLocalToWorld");
        static readonly int CubeWorldToLocalId = Shader.PropertyToID("_CubeWorldToLocal");
        static readonly int HasCubeId = Shader.PropertyToID("_HasCube");
        static readonly int FloorPosId = Shader.PropertyToID("_FloorPos");
        static readonly int FloorSizeId = Shader.PropertyToID("_FloorSize");
        static readonly int PlanarFloorEnabledId = Shader.PropertyToID("_PlanarFloorEnabled");
        static readonly int SsrEnabledId = Shader.PropertyToID("_SsrEnabled");
        static readonly int SsrStepsId = Shader.PropertyToID("_SsrSteps");
        static readonly int SsrMaxDistanceId = Shader.PropertyToID("_SsrMaxDistance");
        static readonly int SsrStrideId = Shader.PropertyToID("_SsrStride");
        static readonly int SsrThicknessId = Shader.PropertyToID("_SsrThickness");
        static readonly int TileCol1Id = Shader.PropertyToID("_TileCol1");
        static readonly int TileCol2Id = Shader.PropertyToID("_TileCol2");
        static readonly int TileCol3Id = Shader.PropertyToID("_TileCol3");
        static readonly int TileCol4Id = Shader.PropertyToID("_TileCol4");
        static readonly int TileColVariationId = Shader.PropertyToID("_TileColVariation");
        static readonly int TileScaleId = Shader.PropertyToID("_TileScale");
        static readonly int TileDarkOffsetId = Shader.PropertyToID("_TileDarkOffset");
        static readonly int TileOriginId = Shader.PropertyToID("_TileOrigin");
        static readonly int DiffusePositionsId = Shader.PropertyToID("_DiffusePositions");
        static readonly int DiffuseVelocitiesId = Shader.PropertyToID("_DiffuseVelocities");
        static readonly int DiffuseLifeId = Shader.PropertyToID("_DiffuseLife");
        static readonly int DiffuseKindId = Shader.PropertyToID("_DiffuseKind");
        static readonly int DiffuseOccupiedId = Shader.PropertyToID("_DiffuseOccupied");
        static readonly int DiffuseRadiusId = Shader.PropertyToID("_DiffuseRadius");
        static readonly int FluidShadowMapId = Shader.PropertyToID("_FluidShadowMap");
        static readonly int FluidShadowVpId = Shader.PropertyToID("_FluidShadowVP");
        static readonly int FluidShadowExtinctionId = Shader.PropertyToID("_FluidShadowExtinction");
        static readonly int FluidShadowAmbientId = Shader.PropertyToID("_FluidShadowAmbient");

        static readonly int DepthRtId = Shader.PropertyToID("_FluidDepthA");
        static readonly int ThicknessRtId = Shader.PropertyToID("_FluidThicknessRT");
        static readonly int PackedAId = Shader.PropertyToID("_FluidPackedA");
        static readonly int PackedBId = Shader.PropertyToID("_FluidPackedB");
        static readonly int SceneColorRtId = Shader.PropertyToID("_FluidSceneColorRT");
        static readonly int SceneBlurAId = Shader.PropertyToID("_FluidSceneBlurA");
        static readonly int SceneBlurBId = Shader.PropertyToID("_FluidSceneBlurB");
        static readonly int FoamRtId = Shader.PropertyToID("_FluidFoamRT");

        [SerializeField]
        FluidSimulation simulation;

        [SerializeField]
        Shader depthShader;

        [SerializeField]
        Shader thicknessShader;

        [SerializeField]
        Shader filterShader;

        [SerializeField]
        Shader compositeShader;

        [SerializeField]
        Shader diffuseShader;

        [SerializeField]
        Shader foamDepthCopyShader;

        [SerializeField]
        Shader shadowBlurShader;

        [SerializeField]
        Shader sceneBlurShader;

        [SerializeField]
        ComputeShader densityVolumeShader2D;

        [SerializeField]
        ComputeShader densityVolumeShader3D;

        [Header("Surface")]
        [Tooltip("Drawn depth-impostor radius as a multiple of the physical particle radius.")]
        [SerializeField, FormerlySerializedAs("radiusScale"), Range(0.8f, 4f)]
        float depthRadiusScale = 2.4f;

        [Tooltip("Thickness-impostor radius as a multiple of the physical particle radius.")]
        [SerializeField, Range(0.8f, 4f)]
        float thicknessRadiusScale = 2.4f;

        [SerializeField]
        FluidThicknessMode thicknessMode = FluidThicknessMode.OverlapCount;

        [Tooltip("OverlapCount: each covering disc adds this constant, matching SebLague.")]
        [SerializeField, Min(0f)]
        float thicknessContribution = 0.1f;

        [SerializeField, Range(0.25f, 1f)]
        float resolutionScale = 1f;

        [SerializeField]
        bool useHalfResThickness = true;

        [Header("Filter")]
        [Tooltip("FluidRenderTest BlurType. Bilateral keeps silhouettes; Gaussian " +
                 "is the smoother sheet.")]
        [SerializeField]
        FluidSurfaceSmoothType surfaceSmooth = FluidSurfaceSmoothType.Bilateral;

        [Tooltip("World-space blur radius in metres. 0 uses the SPH support radius.")]
        [SerializeField, Min(0f)]
        float worldFilterRadius;

        [Tooltip("Pixel cap for the per-pixel kernel. 0 uses 24.")]
        [SerializeField, FormerlySerializedAs("filterRadius"), Range(0, 32)]
        int maxScreenSpaceFilterRadius;

        [SerializeField, Range(1, 8)]
        int filterIterations = 4;

        [Tooltip("Smaller values blur more. Kernel sigma is radius / (6 * strength).")]
        [SerializeField, Min(0.05f)]
        float filterStrength = 0.85f;

        [Tooltip("Camera-distance difference, in metres, that still blends. About one support radius.")]
        [SerializeField, Min(1e-3f)]
        float depthThreshold = 0.12f;

        [Header("Background")]
        [Tooltip("Gaussian blur of the grabbed scene, mixed in by thickness so the " +
                 "floor and box stay readable through the water. 0 is a sharp grab.")]
        [SerializeField, Min(0f)]
        float backgroundBlur = 1.2f;

        [Tooltip("Separable GaussSmooth iterations on a half-res grab, matching " +
                 "FluidRenderTest's shadow / surface blur passes.")]
        [SerializeField, Range(1, 4)]
        int sceneBlurIterations = 2;

        [Tooltip("Keeps a fraction of the tank grab even when absorption is strong, " +
                 "so the cube and floor stay visible inside the volume.")]
        [SerializeField, Range(0f, 1f)]
        float backgroundReveal = 0.12f;

        [Tooltip("How much of the straight-behind tank (floor, box, glass) shows " +
                 "through versus the refracted view. Higher is more transparent.")]
        [SerializeField, Range(0f, 1f)]
        float interiorMix = 0.4f;

        [Header("Volume")]
        [Tooltip("Seb RayMarchingTest extinctionCoeff. RGB absorption per metre of " +
                 "optical depth. Red dies first in water.")]
        [SerializeField, FormerlySerializedAs("absorbColor")]
        Color absorptionCoefficient = new Color(0.55f, 0.16f, 0.07f, 1f);

        [Tooltip("Seb RayMarchingTest densityMultiplier. Scales density along the " +
                 "marched path: optical += (density - offset) * this * stepSize. " +
                 "Bound as this / restDensity, matching RayMarchingTest's / 1000.")]
        [SerializeField, Min(0f)]
        float densityMultiplier = 2f;

        [Tooltip("Seb RayMarchingTest densityOffset / volumeValueOffset. Density " +
                 "below this is air, so overlapping particles fuse into one surface.")]
        [SerializeField, Min(0f)]
        float densityOffset = 150f;

        [Tooltip("Longest-axis resolution of the SPH density Texture3D.")]
        [SerializeField, Range(32, 160)]
        int densityTextureRes = 96;

        [Tooltip("Seb RayMarchingTest numRefractions. Each bounce traces the more " +
                 "interesting of reflect vs refract through the volume.")]
        [SerializeField, Range(1, 8)]
        int numRefractions = 4;

        [SerializeField, Min(1f)]
        float indexOfRefraction = 1.33f;

        [Tooltip("Coarser step for optical-depth probes along the discarded bounce.")]
        [SerializeField, Min(0.05f)]
        float lightStepSize = 0.4f;

        [SerializeField]
        Color scatterColor = new Color(0.02f, 0.18f, 0.28f, 1f);

        [Header("Shading")]
        [Tooltip("Scales the grab-sample offset along the marched exit. 1 uses the " +
                 "real floor / box hit. Smaller values flatten the refraction.")]
        [SerializeField, Min(0f)]
        float refractionMultiplier = 1f;

        [SerializeField, Min(1f)]
        float specularPower = 96f;

        [SerializeField, Range(0f, 4f)]
        float specularIntensity = 0.7f;

        [Tooltip("Scale on the physical air/water Fresnel term. Lower keeps the " +
                 "camera looking into the volume instead of a sky mirror.")]
        [SerializeField, Range(0f, 1f)]
        float fresnelStrength = 0.4f;

        [SerializeField, Min(0f)]
        float thicknessScale = 0.4f;

        [SerializeField, HideInInspector]
        float extinctionMultiplier = 0.4f;

        [Tooltip("Metres per sample along the refracted ray. Seb RayMarchingTest stepSize.")]
        [SerializeField, Min(0.02f)]
        float rayMarchStep = 0.02f;

        [SerializeField, Range(8, 64)]
        int rayMarchSteps = 48;

        [SerializeField]
        FluidDebugView debugView = FluidDebugView.Composite;

        [Header("Reflection")]
        [Tooltip("Shade the tiled floor at the reflected ray's world hit, even when " +
                 "that point is off-screen. This is Seb's planar environment, not a cubemap.")]
        [SerializeField]
        bool planarFloor = true;

        [Tooltip("March the grabbed scene depth along the reflection ray so the rigid " +
                 "box and tank walls appear in the water.")]
        [SerializeField]
        bool screenSpaceReflection = true;

        [SerializeField, Range(8, 64)]
        int ssrSteps = 32;

        [SerializeField, Min(1f)]
        float ssrMaxDistance = 16f;

        [SerializeField, Min(0.02f)]
        float ssrStride = 0.08f;

        [Tooltip("How far behind scene depth a march sample may be and still count as a hit.")]
        [SerializeField, Min(0.05f)]
        float ssrThickness = 0.55f;

        [Header("Shadow")]
        [Tooltip("Render fluid thickness from the sun and darken the tiled floor.")]
        [SerializeField]
        bool castFluidShadow = true;

        [SerializeField, Range(128, 1024)]
        int shadowMapSize = 512;

        [SerializeField, Range(1, 4)]
        int shadowBlurIterations = 2;

        [Tooltip("Scales Beer-Lambert extinction on the floor. Lower keeps tiles readable.")]
        [SerializeField, Min(0f)]
        float shadowExtinctionScale = 0.35f;

        [SerializeField, Range(0f, 0.5f)]
        float shadowAmbient = 0.17f;

        [Header("Diffuse")]
        [SerializeField, Range(0.2f, 8f)]
        float diffuseRadiusScale = 3f;

        readonly Dictionary<Camera, CommandBuffer> buffers = new Dictionary<Camera, CommandBuffer>();

        Material depthMaterial;
        Material thicknessMaterial;
        Material filterMaterial;
        Material compositeMaterial;
        Material diffuseMaterial;
        Material foamDepthCopyMaterial;
        Material thicknessShadowMaterial;
        Material shadowBlurMaterial;
        Material sceneBlurMaterial;

        LocalKeyword depthKeyword2D;
        LocalKeyword depthKeyword3D;
        LocalKeyword thicknessKeyword2D;
        LocalKeyword thicknessKeyword3D;
        LocalKeyword thicknessShadowKeyword;

        CommandBuffer shadowCommand;
        RenderTexture shadowRt;
        RenderTexture shadowBlurRt;
        Texture2D blackTexture;
        readonly FluidDensityMap densityMap = new FluidDensityMap();
        int densityFrame = -1;

        void OnEnable()
        {
            if (simulation == null)
            {
                simulation = GetComponent<FluidSimulation>();
            }

            if (!CreateMaterials())
            {
                enabled = false;
                return;
            }

            Camera.onPreRender += OnCameraPreRender;
        }

        void OnDisable()
        {
            Camera.onPreRender -= OnCameraPreRender;
            RemoveAllBuffers();
            ReleaseShadowResources();
            densityMap.Dispose();
            DestroyMaterial(ref depthMaterial);
            DestroyMaterial(ref thicknessMaterial);
            DestroyMaterial(ref filterMaterial);
            DestroyMaterial(ref compositeMaterial);
            DestroyMaterial(ref diffuseMaterial);
            DestroyMaterial(ref foamDepthCopyMaterial);
            DestroyMaterial(ref thicknessShadowMaterial);
            DestroyMaterial(ref shadowBlurMaterial);
            DestroyMaterial(ref sceneBlurMaterial);
            if (blackTexture != null)
            {
                DestroyImmediate(blackTexture);
                blackTexture = null;
            }
        }

        void OnCameraPreRender(Camera camera)
        {
            if (camera == null || camera.cameraType != CameraType.Game)
            {
                return;
            }

            if (simulation == null || !simulation.IsReady ||
                depthMaterial == null || thicknessMaterial == null ||
                filterMaterial == null || compositeMaterial == null)
            {
                RemoveBuffer(camera);
                ClearFluidShadowGlobals();
                return;
            }

            camera.depthTextureMode |= DepthTextureMode.Depth;
            UpdateDensityMap();
            RenderFluidShadow(camera);

            if (!buffers.TryGetValue(camera, out CommandBuffer command))
            {
                command = new CommandBuffer { name = "FluidSim Screen-Space Fluid" };
                camera.AddCommandBuffer(InjectionPoint, command);
                buffers[camera] = command;
            }

            command.Clear();
            Record(command, camera);
        }

        void Record(CommandBuffer command, Camera camera)
        {
            FluidParameters parameters = simulation.Parameters;
            int particleCount = simulation.Particles.Count;
            int width = Mathf.Max(8, Mathf.RoundToInt(camera.pixelWidth * resolutionScale));
            int height = Mathf.Max(8, Mathf.RoundToInt(camera.pixelHeight * resolutionScale));
            int thicknessWidth = useHalfResThickness ? Mathf.Max(8, width / 2) : width;
            int thicknessHeight = useHalfResThickness ? Mathf.Max(8, height / 2) : height;
            float depthRadius = parameters.ParticleRadius * depthRadiusScale;
            float thicknessRadius = parameters.ParticleRadius * thicknessRadiusScale;
            bool isThreeDimensional = parameters.Dimension == SimulationDimension.Three;

            BindParticlePass(depthMaterial, depthKeyword2D, depthKeyword3D, isThreeDimensional, depthRadius, parameters);
            BindParticlePass(thicknessMaterial, thicknessKeyword2D, thicknessKeyword3D, isThreeDimensional,
                thicknessRadius, parameters);
            thicknessMaterial.SetFloat(ThicknessContributionId, thicknessContribution);
            thicknessMaterial.SetInt(ThicknessModeId, (int)thicknessMode);
            BindShading(command, camera);
            BindFilter(command, camera, parameters);

            command.GetTemporaryRT(DepthRtId, width, height, 24, FilterMode.Point, RenderTextureFormat.RFloat);
            command.GetTemporaryRT(ThicknessRtId, thicknessWidth, thicknessHeight, 16, FilterMode.Bilinear,
                RenderTextureFormat.RHalf);
            command.GetTemporaryRT(PackedAId, width, height, 0, FilterMode.Bilinear, RenderTextureFormat.ARGBFloat);
            command.GetTemporaryRT(PackedBId, width, height, 0, FilterMode.Bilinear, RenderTextureFormat.ARGBFloat);
            command.GetTemporaryRT(SceneColorRtId, camera.pixelWidth, camera.pixelHeight, 0, FilterMode.Bilinear,
                RenderTextureFormat.DefaultHDR);
            command.GetTemporaryRT(FoamRtId, width, height, 16, FilterMode.Bilinear, RenderTextureFormat.ARGBFloat);

            command.Blit(BuiltinRenderTextureType.CameraTarget, SceneColorRtId);
            bool blurredScene = BlurSceneGrab(command, camera);

            DrawFoamBuffer(command, parameters);

            command.SetRenderTarget(DepthRtId);
            command.ClearRenderTarget(true, true, new Color(EmptyDepth, EmptyDepth, EmptyDepth, EmptyDepth));
            command.DrawProcedural(
                Matrix4x4.identity, depthMaterial, 0, MeshTopology.Triangles,
                particleCount * VerticesPerParticle);

            command.SetRenderTarget(ThicknessRtId);
            command.ClearRenderTarget(true, true, Color.black);
            if (foamDepthCopyMaterial != null)
            {
                command.Blit(FoamRtId, ThicknessRtId, foamDepthCopyMaterial);
            }

            command.SetRenderTarget(ThicknessRtId);
            command.DrawProcedural(
                Matrix4x4.identity, thicknessMaterial, 0, MeshTopology.Triangles,
                particleCount * VerticesPerParticle);

            command.SetGlobalTexture(FluidRawDepthId, DepthRtId);
            command.SetGlobalTexture(FluidRawThicknessId, ThicknessRtId);
            command.Blit(DepthRtId, PackedAId, filterMaterial, FilterPackPass);

            int source = PackedAId;
            int destination = PackedBId;
            int passes = Mathf.Max(filterIterations, 1);
            for (int i = 0; i < passes; i++)
            {
                command.Blit(source, destination, filterMaterial, FilterHorizontalPass);
                Swap(ref source, ref destination);
                command.Blit(source, destination, filterMaterial, FilterVerticalPass);
                Swap(ref source, ref destination);
            }

            command.SetGlobalTexture(FluidSceneColorId, SceneColorRtId);
            command.SetGlobalTexture(FluidFoamId, FoamRtId);
            command.Blit(source, BuiltinRenderTextureType.CameraTarget, compositeMaterial);

            command.ReleaseTemporaryRT(DepthRtId);
            command.ReleaseTemporaryRT(ThicknessRtId);
            command.ReleaseTemporaryRT(PackedAId);
            command.ReleaseTemporaryRT(PackedBId);
            command.ReleaseTemporaryRT(SceneColorRtId);
            if (blurredScene)
            {
                command.ReleaseTemporaryRT(SceneBlurAId);
                command.ReleaseTemporaryRT(SceneBlurBId);
            }

            command.ReleaseTemporaryRT(FoamRtId);
        }

        bool BlurSceneGrab(CommandBuffer command, Camera camera)
        {
            if (sceneBlurMaterial == null || backgroundBlur <= 1e-4f)
            {
                command.SetGlobalTexture(FluidSceneBlurId, SceneColorRtId);
                return false;
            }

            int width = Mathf.Max(8, camera.pixelWidth / 2);
            int height = Mathf.Max(8, camera.pixelHeight / 2);
            command.GetTemporaryRT(SceneBlurAId, width, height, 0, FilterMode.Bilinear, RenderTextureFormat.DefaultHDR);
            command.GetTemporaryRT(SceneBlurBId, width, height, 0, FilterMode.Bilinear, RenderTextureFormat.DefaultHDR);

            command.Blit(SceneColorRtId, SceneBlurAId, sceneBlurMaterial, 0);
            command.Blit(SceneBlurAId, SceneBlurBId, sceneBlurMaterial, 1);
            int iterations = Mathf.Max(sceneBlurIterations, 1);
            for (int i = 1; i < iterations; i++)
            {
                command.Blit(SceneBlurBId, SceneBlurAId, sceneBlurMaterial, 0);
                command.Blit(SceneBlurAId, SceneBlurBId, sceneBlurMaterial, 1);
            }

            command.SetGlobalTexture(FluidSceneBlurId, SceneBlurBId);
            return true;
        }

        void RenderFluidShadow(Camera camera)
        {
            if (!castFluidShadow || thicknessShadowMaterial == null || simulation == null ||
                !simulation.IsReady)
            {
                ClearFluidShadowGlobals();
                return;
            }

            FluidParameters parameters = simulation.Parameters;
            EnsureShadowTargets(shadowMapSize);
            if (shadowRt == null)
            {
                ClearFluidShadowGlobals();
                return;
            }

            Light sun = RenderSettings.sun;
            Vector3 lightDir = sun != null
                ? sun.transform.forward
                : new Vector3(0.35f, -0.85f, 0.4f).normalized;
            Quaternion rotation = sun != null
                ? sun.transform.rotation
                : Quaternion.LookRotation(lightDir, Vector3.up);
            Bounds domain = parameters.Domain;
            float distance = domain.extents.magnitude + 2f;
            Vector3 position = domain.center - lightDir * distance;
            Matrix4x4 worldToCamera = WorldToCameraMatrix(position, rotation);
            float ortho = FrameBoundsOrtho(domain, worldToCamera) + 0.5f;
            float farClip = distance + domain.extents.magnitude + 2f;
            Matrix4x4 projection = Matrix4x4.Ortho(-ortho, ortho, -ortho, ortho, 0.05f, farClip);
            Matrix4x4 viewProjection = GL.GetGPUProjectionMatrix(projection, true) * worldToCamera;

            bool isThreeDimensional = parameters.Dimension == SimulationDimension.Three;
            BindParticlePass(
                thicknessShadowMaterial, thicknessKeyword2D, thicknessKeyword3D,
                isThreeDimensional, parameters.ParticleRadius * thicknessRadiusScale, parameters);
            thicknessShadowMaterial.SetFloat(ThicknessContributionId, thicknessContribution);
            thicknessShadowMaterial.SetInt(ThicknessModeId, (int)thicknessMode);
            thicknessShadowMaterial.SetKeyword(thicknessShadowKeyword, true);

            if (shadowCommand == null)
            {
                shadowCommand = new CommandBuffer { name = "FluidSim Thickness Shadow" };
            }

            shadowCommand.Clear();
            shadowCommand.SetViewProjectionMatrices(worldToCamera, projection);
            shadowCommand.SetRenderTarget(shadowRt);
            shadowCommand.ClearRenderTarget(true, true, Color.black);
            shadowCommand.DrawProcedural(
                Matrix4x4.identity, thicknessShadowMaterial, 1, MeshTopology.Triangles,
                simulation.Particles.Count * VerticesPerParticle);

            Graphics.ExecuteCommandBuffer(shadowCommand);
            camera.ResetWorldToCameraMatrix();
            camera.ResetProjectionMatrix();

            if (shadowBlurMaterial != null && shadowBlurRt != null)
            {
                shadowCommand.Clear();
                int iterations = Mathf.Max(shadowBlurIterations, 1);
                for (int i = 0; i < iterations; i++)
                {
                    shadowCommand.Blit(shadowRt, shadowBlurRt, shadowBlurMaterial, 0);
                    shadowCommand.Blit(shadowBlurRt, shadowRt, shadowBlurMaterial, 1);
                }

                Graphics.ExecuteCommandBuffer(shadowCommand);
            }

            Vector3 extinction = new Vector3(
                absorptionCoefficient.r, absorptionCoefficient.g, absorptionCoefficient.b) *
                                 densityMultiplier * shadowExtinctionScale;
            Shader.SetGlobalTexture(FluidShadowMapId, shadowRt);
            Shader.SetGlobalMatrix(FluidShadowVpId, viewProjection);
            Shader.SetGlobalVector(FluidShadowExtinctionId, extinction);
            Shader.SetGlobalFloat(FluidShadowAmbientId, shadowAmbient);
        }

        void ClearFluidShadowGlobals()
        {
            Texture fallback = blackTexture != null ? (Texture)blackTexture : Texture2D.blackTexture;
            Shader.SetGlobalTexture(FluidShadowMapId, fallback);
            Shader.SetGlobalMatrix(FluidShadowVpId, Matrix4x4.identity);
            Shader.SetGlobalVector(FluidShadowExtinctionId, Vector3.zero);
            Shader.SetGlobalFloat(FluidShadowAmbientId, 1f);
        }

        void EnsureShadowTargets(int size)
        {
            size = Mathf.Clamp(size, 128, 1024);
            if (shadowRt != null && shadowRt.width == size)
            {
                return;
            }

            ReleaseShadowTargets();
            shadowRt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat)
            {
                name = "FluidSim Shadow",
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                hideFlags = HideFlags.HideAndDontSave
            };
            shadowRt.Create();
            shadowBlurRt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat)
            {
                name = "FluidSim Shadow Blur",
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                hideFlags = HideFlags.HideAndDontSave
            };
            shadowBlurRt.Create();
        }

        void ReleaseShadowTargets()
        {
            if (shadowRt != null)
            {
                shadowRt.Release();
                DestroyImmediate(shadowRt);
                shadowRt = null;
            }

            if (shadowBlurRt != null)
            {
                shadowBlurRt.Release();
                DestroyImmediate(shadowBlurRt);
                shadowBlurRt = null;
            }
        }

        void ReleaseShadowResources()
        {
            if (shadowCommand != null)
            {
                shadowCommand.Release();
                shadowCommand = null;
            }

            ReleaseShadowTargets();
            ClearFluidShadowGlobals();
        }

        static Matrix4x4 WorldToCameraMatrix(Vector3 position, Quaternion rotation)
        {
            Matrix4x4 worldToCamera = Matrix4x4.TRS(position, rotation, Vector3.one).inverse;
            worldToCamera.m20 = -worldToCamera.m20;
            worldToCamera.m21 = -worldToCamera.m21;
            worldToCamera.m22 = -worldToCamera.m22;
            worldToCamera.m23 = -worldToCamera.m23;
            return worldToCamera;
        }

        static float FrameBoundsOrtho(Bounds domain, Matrix4x4 worldToView)
        {
            float maxX = 0f;
            float maxY = 0f;
            Vector3 min = domain.min;
            Vector3 size = domain.size;
            for (int i = 0; i < 8; i++)
            {
                Vector3 world = min + Vector3.Scale(size, new Vector3(
                    (i & 1) == 0 ? 0f : 1f,
                    (i & 2) == 0 ? 0f : 1f,
                    (i & 4) == 0 ? 0f : 1f));
                Vector3 view = worldToView.MultiplyPoint(world);
                maxX = Mathf.Max(maxX, Mathf.Abs(view.x));
                maxY = Mathf.Max(maxY, Mathf.Abs(view.y));
            }

            return Mathf.Max(maxX, maxY);
        }

        void UpdateDensityMap()
        {
            if (densityFrame == Time.frameCount && densityMap.Texture != null)
            {
                return;
            }

            densityFrame = Time.frameCount;
            ComputeShader shader = simulation.Parameters.Dimension == SimulationDimension.Two
                ? densityVolumeShader2D
                : densityVolumeShader3D;
            densityMap.SetShader(shader);
            densityMap.Update(simulation, densityTextureRes);
        }

        void BindVolumeEnvironment(Material material)
        {
            Bounds domain = simulation.Parameters.Domain;
            bool hasCube = false;
            Matrix4x4 cubeLocalToWorld = Matrix4x4.identity;
            if (simulation.RigidBody != null && simulation.SpawnDemoRigidBox)
            {
                Vector3 center = (Vector3)simulation.RigidBody.Position;
                Quaternion rotation = (Quaternion)simulation.RigidBody.Rotation;
                Vector3 size = (Vector3)simulation.RigidBody.Size;
                cubeLocalToWorld = Matrix4x4.TRS(center, rotation, size * 0.5f);
                hasCube = size.sqrMagnitude > 1e-6f;
            }
            else if (simulation.SpawnDemoRigidBox)
            {
                cubeLocalToWorld = Matrix4x4.TRS(
                    simulation.DemoRigidBoxCenter, Quaternion.identity, simulation.DemoRigidBoxSize * 0.5f);
                hasCube = true;
            }

            var demo = GetComponent<FluidDemoVisuals>();
            Vector3 floorCenter;
            Vector3 floorFullSize;
            if (demo != null)
            {
                demo.GetFloorBounds(domain, out floorCenter, out floorFullSize);
                demo.ApplyTiles(material);
            }
            else
            {
                FluidDemoVisuals.DefaultFloorBounds(domain, 24f, out floorCenter, out floorFullSize);
                material.SetColor(TileCol1Id, new Color(0.82f, 0.78f, 0.72f, 1f));
                material.SetColor(TileCol2Id, new Color(0.76f, 0.80f, 0.84f, 1f));
                material.SetColor(TileCol3Id, new Color(0.80f, 0.74f, 0.70f, 1f));
                material.SetColor(TileCol4Id, new Color(0.72f, 0.76f, 0.78f, 1f));
                material.SetVector(TileColVariationId, new Vector3(0.02f, 0.05f, 0.06f));
                material.SetFloat(TileScaleId, 1.5f);
                material.SetFloat(TileDarkOffsetId, -0.18f);
                material.SetVector(TileOriginId, domain.center);
            }

            material.SetMatrix(CubeLocalToWorldId, cubeLocalToWorld);
            material.SetMatrix(CubeWorldToLocalId, cubeLocalToWorld.inverse);
            material.SetFloat(HasCubeId, hasCube ? 1f : 0f);
            material.SetVector(FloorPosId, floorCenter);
            material.SetVector(FloorSizeId, floorFullSize * 0.5f);
            material.SetFloat(PlanarFloorEnabledId, planarFloor ? 1f : 0f);
            material.SetFloat(SsrEnabledId, screenSpaceReflection ? 1f : 0f);
            material.SetInt(SsrStepsId, ssrSteps);
            material.SetFloat(SsrMaxDistanceId, ssrMaxDistance);
            material.SetFloat(SsrStrideId, ssrStride);
            material.SetFloat(SsrThicknessId, ssrThickness);
        }

        void BindFilter(CommandBuffer command, Camera camera, FluidParameters parameters)
        {
            float worldRadius = worldFilterRadius > 1e-5f ? worldFilterRadius : parameters.SupportRadius;
            int maxPixels = maxScreenSpaceFilterRadius > 0 ? maxScreenSpaceFilterRadius : 24;
            float rangeSigma = Mathf.Max(depthThreshold, 1e-4f);
            float projectionM00 = GL.GetGPUProjectionMatrix(camera.projectionMatrix, true).m00;

            command.SetGlobalFloat(WorldFilterRadiusId, worldRadius);
            command.SetGlobalInt(MaxScreenSpaceRadiusId, maxPixels);
            command.SetGlobalFloat(FilterStrengthId, filterStrength);
            command.SetGlobalFloat(FilterDiffStrengthId, 1f / (rangeSigma * rangeSigma));
            float bilateral = surfaceSmooth == FluidSurfaceSmoothType.Bilateral ? 1f : 0f;
            command.SetGlobalFloat(FilterBilateralId, bilateral);
            filterMaterial.SetFloat(FilterBilateralId, bilateral);
            command.SetGlobalFloat(CameraProjectionM00Id, projectionM00);
        }

        void BindParticlePass(
            Material material,
            LocalKeyword keyword2D,
            LocalKeyword keyword3D,
            bool isThreeDimensional,
            float radius,
            FluidParameters parameters)
        {
            material.SetKeyword(keyword3D, isThreeDimensional);
            material.SetKeyword(keyword2D, !isThreeDimensional);
            material.SetBuffer(PositionsId, simulation.Particles.SortedPositions);
            material.SetFloat(ParticleRadiusId, radius);
            material.SetFloat(PlaneDepthId, parameters.Domain.center.z);
        }

        void BindShading(CommandBuffer command, Camera camera)
        {
            Vector3 worldLight = RenderSettings.sun != null
                ? -RenderSettings.sun.transform.forward
                : new Vector3(0.35f, 0.85f, -0.4f).normalized;
            Vector3 viewLight = camera.worldToCameraMatrix.MultiplyVector(worldLight);
            if (viewLight.sqrMagnitude < 1e-8f)
            {
                viewLight = new Vector3(0f, 0f, 1f);
            }

            Bounds domain = simulation.Parameters.Domain;
            Matrix4x4 gpuProjection = GL.GetGPUProjectionMatrix(camera.projectionMatrix, true);
            Matrix4x4 viewProjection = gpuProjection * camera.worldToCameraMatrix;

            command.SetGlobalVector(LightDirectionVsId, viewLight.normalized);
            command.SetGlobalVector(DomainCenterId, domain.center);
            command.SetGlobalVector(DomainSizeId, domain.size);
            command.SetGlobalMatrix(FluidCameraVpId, viewProjection);
            command.SetGlobalMatrix(FluidInvPId, gpuProjection.inverse);
            command.SetGlobalMatrix(FluidCameraToWorldId, camera.cameraToWorldMatrix);
            command.SetGlobalMatrix(FluidWorldToCameraId, camera.worldToCameraMatrix);
            command.SetGlobalFloat(RefractionMultiplierId, refractionMultiplier);
            float restDensity = Mathf.Max(simulation.Parameters.RestDensity, 1f);
            float scaledDensityMultiplier = densityMultiplier / restDensity;
            command.SetGlobalFloat(DensityMultiplierId, scaledDensityMultiplier);
            command.SetGlobalFloat(DensityOffsetId, densityOffset);
            command.SetGlobalFloat(LightStepSizeId, lightStepSize);
            command.SetGlobalInt(NumRefractionsId, numRefractions);
            command.SetGlobalFloat(IndexOfRefractionId, indexOfRefraction);
            command.SetGlobalVector(DirToSunId, worldLight.normalized);
            command.SetGlobalFloat(RayMarchStepId, rayMarchStep);
            command.SetGlobalInt(RayMarchStepsId, rayMarchSteps);
            compositeMaterial.SetColor(AbsorbColorId, absorptionCoefficient);
            compositeMaterial.SetColor(ScatterColorId, scatterColor);
            compositeMaterial.SetFloat(RefractionMultiplierId, refractionMultiplier);
            compositeMaterial.SetFloat(DensityMultiplierId, scaledDensityMultiplier);
            compositeMaterial.SetFloat(DensityOffsetId, densityOffset);
            compositeMaterial.SetFloat(LightStepSizeId, lightStepSize);
            compositeMaterial.SetFloat(IndexOfRefractionId, indexOfRefraction);
            compositeMaterial.SetInt(NumRefractionsId, numRefractions);
            compositeMaterial.SetVector(DirToSunId, worldLight.normalized);
            BindVolumeEnvironment(compositeMaterial);
            if (densityMap.Texture != null)
            {
                compositeMaterial.SetTexture(DensityMapId, densityMap.Texture);
                command.SetGlobalTexture(DensityMapId, densityMap.Texture);
            }
            compositeMaterial.SetVector(DomainCenterId, domain.center);
            compositeMaterial.SetVector(DomainSizeId, domain.size);
            compositeMaterial.SetMatrix(FluidCameraVpId, viewProjection);
            compositeMaterial.SetMatrix(FluidInvPId, gpuProjection.inverse);
            compositeMaterial.SetMatrix(FluidCameraToWorldId, camera.cameraToWorldMatrix);
            compositeMaterial.SetMatrix(FluidWorldToCameraId, camera.worldToCameraMatrix);
            compositeMaterial.SetFloat(RayMarchStepId, rayMarchStep);
            compositeMaterial.SetInt(RayMarchStepsId, rayMarchSteps);
            compositeMaterial.SetFloat(SpecularPowerId, specularPower);
            compositeMaterial.SetFloat(SpecularIntensityId, specularIntensity);
            compositeMaterial.SetFloat(FresnelStrengthId, fresnelStrength);
            compositeMaterial.SetFloat(ThicknessScaleId, thicknessScale);
            compositeMaterial.SetFloat(BackgroundBlurId, backgroundBlur);
            compositeMaterial.SetFloat(BackgroundRevealId, backgroundReveal);
            compositeMaterial.SetFloat(InteriorMixId, interiorMix);
            compositeMaterial.SetInt(DebugViewId, (int)debugView);
            if (foamDepthCopyMaterial != null)
            {
                foamDepthCopyMaterial.SetMatrix(FluidCameraVpId, viewProjection);
                foamDepthCopyMaterial.SetMatrix(FluidInvPId, gpuProjection.inverse);
                foamDepthCopyMaterial.SetMatrix(FluidCameraToWorldId, camera.cameraToWorldMatrix);
            }
        }

        void DrawFoamBuffer(CommandBuffer command, FluidParameters parameters)
        {
            float farUnityDepth = SystemInfo.usesReversedZBuffer ? 0f : 1f;
            command.SetRenderTarget(FoamRtId);
            command.ClearRenderTarget(true, true, new Color(0f, farUnityDepth, 0f, 0f));

            DiffuseMaterialSystem diffuse = simulation.Diffuse;
            if (diffuseMaterial == null || diffuse == null)
            {
                return;
            }

            diffuseMaterial.SetBuffer(DiffusePositionsId, diffuse.Positions);
            diffuseMaterial.SetBuffer(DiffuseVelocitiesId, diffuse.Velocities);
            diffuseMaterial.SetBuffer(DiffuseLifeId, diffuse.Life);
            diffuseMaterial.SetBuffer(DiffuseKindId, diffuse.Kind);
            diffuseMaterial.SetBuffer(DiffuseOccupiedId, diffuse.Occupied);
            diffuseMaterial.SetFloat(DiffuseRadiusId, parameters.ParticleRadius * diffuseRadiusScale);

            command.DrawProcedural(
                Matrix4x4.identity, diffuseMaterial, 0, MeshTopology.Triangles,
                diffuse.Capacity * VerticesPerParticle);
        }

        bool CreateMaterials()
        {
            Shader depth = depthShader != null ? depthShader : Shader.Find(DepthShaderName);
            Shader thickness = thicknessShader != null ? thicknessShader : Shader.Find(ThicknessShaderName);
            Shader filter = filterShader != null ? filterShader : Shader.Find(FilterShaderName);
            Shader composite = compositeShader != null ? compositeShader : Shader.Find(CompositeShaderName);
            Shader diffuse = diffuseShader != null ? diffuseShader : Shader.Find(DiffuseShaderName);
            Shader foamDepthCopy = foamDepthCopyShader != null ? foamDepthCopyShader : Shader.Find(FoamDepthCopyShaderName);
            Shader blur = shadowBlurShader != null ? shadowBlurShader : Shader.Find(ShadowBlurShaderName);
            Shader sceneBlur = sceneBlurShader != null ? sceneBlurShader : Shader.Find(SceneBlurShaderName);

            if (depth == null || thickness == null || filter == null || composite == null)
            {
                Debug.LogError(
                    "Could not find the screen-space fluid shaders. Assign them explicitly " +
                    "or recreate the object from GameObject > FluidSim > Fluid Simulation.", this);
                return false;
            }

            depthMaterial = CreateMaterial(depth);
            thicknessMaterial = CreateMaterial(thickness);
            thicknessShadowMaterial = CreateMaterial(thickness);
            filterMaterial = CreateMaterial(filter);
            compositeMaterial = CreateMaterial(composite);
            if (diffuse != null)
            {
                diffuseMaterial = CreateMaterial(diffuse);
            }

            if (foamDepthCopy != null)
            {
                foamDepthCopyMaterial = CreateMaterial(foamDepthCopy);
            }

            if (blur != null)
            {
                shadowBlurMaterial = CreateMaterial(blur);
            }

            if (sceneBlur != null)
            {
                sceneBlurMaterial = CreateMaterial(sceneBlur);
            }

            if (blackTexture == null)
            {
                blackTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false)
                {
                    hideFlags = HideFlags.HideAndDontSave,
                    name = "FluidSim Shadow Black"
                };
                blackTexture.SetPixel(0, 0, Color.black);
                blackTexture.Apply();
            }

            depthKeyword2D = new LocalKeyword(depth, TwoDimensionalKeyword);
            depthKeyword3D = new LocalKeyword(depth, ThreeDimensionalKeyword);
            thicknessKeyword2D = new LocalKeyword(thickness, TwoDimensionalKeyword);
            thicknessKeyword3D = new LocalKeyword(thickness, ThreeDimensionalKeyword);
            thicknessShadowKeyword = new LocalKeyword(thickness, ShadowPassKeywordName);
            thicknessShadowMaterial.SetKeyword(thicknessShadowKeyword, true);
            return true;
        }

        static Material CreateMaterial(Shader shader)
        {
            return new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
        }

        static void DestroyMaterial(ref Material material)
        {
            if (material != null)
            {
                DestroyImmediate(material);
                material = null;
            }
        }

        static void Swap(ref int a, ref int b)
        {
            int tmp = a;
            a = b;
            b = tmp;
        }

        void RemoveBuffer(Camera camera)
        {
            if (camera == null || !buffers.TryGetValue(camera, out CommandBuffer command))
            {
                return;
            }

            camera.RemoveCommandBuffer(InjectionPoint, command);
            command.Release();
            buffers.Remove(camera);
        }

        void RemoveAllBuffers()
        {
            foreach (KeyValuePair<Camera, CommandBuffer> pair in buffers)
            {
                if (pair.Key != null)
                {
                    pair.Key.RemoveCommandBuffer(InjectionPoint, pair.Value);
                }

                pair.Value.Release();
            }

            buffers.Clear();
        }
    }
}
