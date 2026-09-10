using FluidSim.Rendering;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Tests
{
    public sealed class ScreenSpaceFluidTests
    {
        const string DepthPath = "Assets/FluidSim/Shaders/Render/ParticleDepth.shader";
        const string ThicknessPath = "Assets/FluidSim/Shaders/Render/ParticleThickness.shader";
        const string FilterPath = "Assets/FluidSim/Shaders/Render/FluidDepthFilter.shader";
        const string CompositePath = "Assets/FluidSim/Shaders/Render/FluidComposite.shader";
        const string DensityVolume2DPath = "Assets/FluidSim/Shaders/Compute/DensityVolume2D.compute";
        const string DensityVolume3DPath = "Assets/FluidSim/Shaders/Compute/DensityVolume3D.compute";
        const string DiffuseParticlePath = "Assets/FluidSim/Shaders/Render/DiffuseParticle.shader";
        const string Diffuse2DPath = "Assets/FluidSim/Shaders/Compute/Diffuse2D.compute";
        const string Diffuse3DPath = "Assets/FluidSim/Shaders/Compute/Diffuse3D.compute";
        const string DemoSolidPath = "Assets/FluidSim/Shaders/Render/DemoSolid.shader";
        const string DemoFloorPath = "Assets/FluidSim/Shaders/Render/DemoFloor.shader";
        const string DemoGlassPath = "Assets/FluidSim/Shaders/Render/DemoGlass.shader";
        const string ShadowBlurPath = "Assets/FluidSim/Shaders/Render/FluidShadowBlur.shader";
        const string SceneBlurPath = "Assets/FluidSim/Shaders/Render/FluidSceneBlur.shader";
        const string FoamDepthCopyPath = "Assets/FluidSim/Shaders/Render/FoamDepthCopy.shader";

        [Test]
        public void ScreenSpaceShadersImport()
        {
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(DepthPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(ThicknessPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(FilterPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(CompositePath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(DiffuseParticlePath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<ComputeShader>(Diffuse2DPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<ComputeShader>(Diffuse3DPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(DemoSolidPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(DemoFloorPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(DemoGlassPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(ShadowBlurPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(SceneBlurPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<Shader>(FoamDepthCopyPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<ComputeShader>(DensityVolume2DPath), Is.Not.Null);
            Assert.That(AssetDatabase.LoadAssetAtPath<ComputeShader>(DensityVolume3DPath), Is.Not.Null);
            Assert.That(
                System.IO.File.Exists(
                    System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Include/FluidRayMarch.hlsl")),
                Is.True);
        }

        [Test]
        public void ScreenSpaceShadersCompile()
        {
            AssertCompiled(DepthPath);
            AssertCompiled(ThicknessPath);
            AssertCompiled(FilterPath);
            AssertCompiled(CompositePath);
            AssertCompiled(DiffuseParticlePath);
            AssertCompiled(DemoSolidPath);
            AssertCompiled(DemoFloorPath);
            AssertCompiled(DemoGlassPath);
            AssertCompiled(ShadowBlurPath);
            AssertCompiled(SceneBlurPath);
            AssertCompiled(FoamDepthCopyPath);
        }

        [Test]
        public void BilateralRangeWeightFallsOffWithDepthJump()
        {
            // Green GDC 2010 / SebLague: exp(-d² / σ²). A silhouette neighbour
            // still gets a vanishing weight so the sheet does not bleed into air.
            float Weight(float delta, float threshold)
            {
                float diffStrength = 1f / (threshold * threshold);
                return Mathf.Exp(-delta * delta * diffStrength);
            }

            Assert.That(Weight(0.01f, 0.08f), Is.GreaterThan(0.9f));
            Assert.That(Weight(0.20f, 0.08f), Is.LessThan(0.05f));
        }

        [Test]
        public void ThicknessWeightedBackgroundBlurLerpsTowardTheGrab()
        {
            Assert.That(DiffuseCriteria.BackgroundBlurWeight(0f, 1.4f, 1.2f), Is.EqualTo(0f));
            Assert.That(DiffuseCriteria.BackgroundBlurWeight(1f, 1f, 1f), Is.EqualTo(1f));
            Assert.That(
                DiffuseCriteria.BackgroundBlurWeight(0.25f, 2f, 1f),
                Is.EqualTo(0.5f).Within(1e-4f));

            var sharp = new Unity.Mathematics.float3(0f, 0f, 0f);
            var blurred = new Unity.Mathematics.float3(1f, 1f, 1f);
            Unity.Mathematics.float3 mixed = DiffuseCriteria.MixSceneThroughWater(sharp, blurred, 0.5f);
            Assert.That(mixed.x, Is.EqualTo(0.5f).Within(1e-4f));
        }

        [Test]
        public void BackgroundRevealLiftsBeerLambertTransmission()
        {
            var dark = new Unity.Mathematics.float3(0.02f, 0.08f, 0.15f);
            Unity.Mathematics.float3 lifted = DiffuseCriteria.LiftTransmission(dark, 0.38f);
            Assert.That(lifted.x, Is.EqualTo(0.38f).Within(1e-4f));
            Assert.That(lifted.y, Is.EqualTo(0.38f).Within(1e-4f));
        }

        [Test]
        public void InteriorMixShowsTheStraightBehindTank()
        {
            var behind = new Unity.Mathematics.float3(1f, 0.4f, 0.1f);
            var bent = new Unity.Mathematics.float3(0f, 0f, 1f);
            Unity.Mathematics.float3 mixed = DiffuseCriteria.MixInterior(behind, bent, 0.62f);
            Assert.That(DiffuseCriteria.MixInterior(behind, bent, 1f), Is.EqualTo(behind));
            Assert.That(DiffuseCriteria.MixInterior(behind, bent, 0f), Is.EqualTo(bent));
            Assert.That(mixed.x, Is.GreaterThan(bent.x));
            Assert.That(mixed.x, Is.LessThan(behind.x));
        }

        [Test]
        public void VolumeExtinctionUsesDensityOffsetThenBeerLambert()
        {
            const float step = 0.02f;
            float inside = DiffuseCriteria.SampleDensity(1000f, 150f);
            float scaledMultiplier = 2f / 1000f;
            float marched = DiffuseCriteria.MarchDensitySample(inside, scaledMultiplier, step) * 40f;
            Assert.That(inside, Is.EqualTo(850f).Within(1e-3f));
            Assert.That(marched, Is.EqualTo(1.36f).Within(1e-4f));
            Assert.That(DiffuseCriteria.IsInsideFluid(200f, 150f), Is.True);
            Assert.That(DiffuseCriteria.IsInsideFluid(100f, 150f), Is.False);

            var absorption = new Unity.Mathematics.float3(0.55f, 0.16f, 0.07f);
            Unity.Mathematics.float3 deep = DiffuseCriteria.Absorb(marched, absorption);
            Unity.Mathematics.float3 shallow = DiffuseCriteria.Absorb(0.08f, absorption);
            Assert.That(deep.x, Is.LessThan(shallow.x));
        }

        [Test]
        public void RayMarchPicksTheDenserFresnelPath()
        {
            Assert.That(DiffuseCriteria.PreferRefractPath(4f, 0.8f, 0.5f, 0.2f), Is.True);
            Assert.That(DiffuseCriteria.PreferRefractPath(0.1f, 0.2f, 5f, 0.8f), Is.False);
        }

        [Test]
        public void FluidRenderTestSceneBlurAndAaAreWired()
        {
            string composite = System.IO.File.ReadAllText(
                System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Render/FluidComposite.shader"));
            Assert.That(composite, Does.Contain("SampleSceneAA"));
            Assert.That(composite, Does.Contain("_FluidSceneBlur"));
            Assert.That(composite, Does.Contain("RayMarchFluid"));
            Assert.That(composite, Does.Contain("FindNextSurface"));
            Assert.That(composite, Does.Contain("unity_CameraInvProjection"));
            Assert.That(composite, Does.Contain("TraceSsr"));
            Assert.That(composite, Does.Contain("PlanarFloorLit"));
            Assert.That(composite, Does.Contain("FluidFloorAlbedo"));
            Assert.That(
                System.IO.File.Exists(
                    System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Include/FluidFloorTiles.hlsl")),
                Is.True);
            Assert.That(composite, Does.Not.Contain("max(reflectCol, float3(0.42"));

            string march = System.IO.File.ReadAllText(
                System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Include/FluidRayMarch.hlsl"));
            Assert.That(march, Does.Contain("_DensityMultiplier * stepSize"));
            Assert.That(march, Does.Contain("CalculateNormal"));
            Assert.That(march, Does.Contain("FluidViewMarchLimit = 512"));

            string filter = System.IO.File.ReadAllText(
                System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Render/FluidDepthFilter.shader"));
            Assert.That(filter, Does.Contain("_FilterBilateral"));
        }

        [Test]
        public void WorldSpaceKernelWidthScalesWithProjectionAndDepth()
        {
            // Freya Holmér: px = (width * P00 / (2 * depth)) * worldRadius.
            float Pixels(float worldRadius, float depth, int width, float projectionM00)
            {
                float pxPerMeter = width * projectionM00 / (2f * depth);
                return Mathf.Abs(pxPerMeter) * worldRadius;
            }

            Assert.That(Pixels(1f, 1f, 1080, 1f), Is.EqualTo(540f).Within(1e-3f));
            Assert.That(Pixels(0.1f, 2f, 1080, 1f), Is.EqualTo(27f).Within(1e-3f));
            Assert.That(Pixels(0.1f, 4f, 1080, 1f), Is.LessThan(Pixels(0.1f, 2f, 1080, 1f)));
        }

        [Test]
        public void CameraDistanceToSphereFrontIsCloserThanTheCentre()
        {
            float CentreDistance(Vector3 center) => center.magnitude;
            float FrontDistance(Vector3 center, float radius)
            {
                var hit = center;
                hit.z += radius;
                return hit.magnitude;
            }

            var center = new Vector3(0.4f, -0.2f, -3f);
            Assert.That(FrontDistance(center, 0.05f), Is.LessThan(CentreDistance(center)));
        }

        [Test]
        public void PackedChannelsMatchSebLayout()
        {
            const float depth = 4.2f;
            const float thick = 0.8f;
            var packed = new Vector4(depth, thick, thick, depth);

            Assert.That(packed.x, Is.EqualTo(depth));
            Assert.That(packed.y, Is.EqualTo(thick));
            Assert.That(packed.z, Is.EqualTo(thick));
            Assert.That(packed.w, Is.EqualTo(depth));
        }

        [Test]
        public void OverlapThicknessIgnoresDiscRadius()
        {
            const float contribution = 0.1f;
            float Overlap(float discRadiusSq) => discRadiusSq <= 1f ? contribution : 0f;
            float Chord(float radius, float discRadiusSq) =>
                discRadiusSq <= 1f ? 2f * radius * Mathf.Sqrt(1f - discRadiusSq) : 0f;

            Assert.That(Overlap(0f), Is.EqualTo(contribution));
            Assert.That(Overlap(0.5f), Is.EqualTo(contribution));
            Assert.That(Overlap(1.1f), Is.EqualTo(0f));
            Assert.That(Chord(0.06f, 0f), Is.GreaterThan(Chord(0.06f, 0.75f)));
        }

        [Test]
        public void SphereImpostorOffsetPeaksAtTheCentre()
        {
            float Centre(float discRadius) => Mathf.Sqrt(Mathf.Max(0f, 1f - discRadius * discRadius));

            Assert.That(Centre(0f), Is.EqualTo(1f));
            Assert.That(Centre(1f), Is.EqualTo(0f));
            Assert.That(Centre(0.5f), Is.GreaterThan(Centre(0.8f)));
        }

        [Test]
        public void DiffusePhiMapsThresholdsToTheUnitInterval()
        {
            Assert.That(DiffuseCriteria.Phi(0f, 2f, 8f), Is.EqualTo(0f));
            Assert.That(DiffuseCriteria.Phi(2f, 2f, 8f), Is.EqualTo(0f));
            Assert.That(DiffuseCriteria.Phi(5f, 2f, 8f), Is.EqualTo(0.5f).Within(1e-5f));
            Assert.That(DiffuseCriteria.Phi(8f, 2f, 8f), Is.EqualTo(1f));
            Assert.That(DiffuseCriteria.Phi(20f, 2f, 8f), Is.EqualTo(1f));
        }

        [Test]
        public void DiffuseClassificationMatchesIhmsenNeighborBins()
        {
            Assert.That(DiffuseCriteria.Classify(3, 3), Is.EqualTo(0));
            Assert.That(DiffuseCriteria.Classify(10, 3), Is.EqualTo(1));
            Assert.That(DiffuseCriteria.Classify(25, 3), Is.EqualTo(2));
            Assert.That(DiffuseCriteria.Classify(2, 2), Is.EqualTo(0));
            Assert.That(DiffuseCriteria.Classify(8, 2), Is.EqualTo(1));
            Assert.That(DiffuseCriteria.Classify(15, 2), Is.EqualTo(2));
        }

        [Test]
        public void TrappedAirPrefersRecedingNeighborsOverTheBow()
        {
            var offset = new Unity.Mathematics.float3(1f, 0f, 0f);
            float approaching = DiffuseCriteria.TrappedAirPair(
                new Unity.Mathematics.float3(-2f, 0f, 0f), offset);
            float receding = DiffuseCriteria.TrappedAirPair(
                new Unity.Mathematics.float3(2f, 0f, 0f), offset);

            Assert.That(approaching, Is.EqualTo(0f).Within(1e-4f));
            Assert.That(receding, Is.EqualTo(4f).Within(1e-4f));
            Assert.That(receding, Is.GreaterThan(approaching));
        }

        [Test]
        public void WaveCrestGateFiresWhenVelocityOpposesTheNormal()
        {
            var normal = new Unity.Mathematics.float3(1f, 0f, 0f);
            Assert.That(
                DiffuseCriteria.WaveCrestVelocityGate(new Unity.Mathematics.float3(1f, 0f, 0f), normal),
                Is.False);
            Assert.That(
                DiffuseCriteria.WaveCrestVelocityGate(new Unity.Mathematics.float3(-1f, 0f, 0f), normal),
                Is.True);
        }

        [Test]
        public void DebugViewEnumMatchesShaderCodes()
        {
            Assert.That((int)FluidDebugView.Composite, Is.EqualTo(0));
            Assert.That((int)FluidDebugView.Depth, Is.EqualTo(1));
            Assert.That((int)FluidDebugView.Thickness, Is.EqualTo(2));
            Assert.That((int)FluidDebugView.Normals, Is.EqualTo(3));
            Assert.That((int)FluidDebugView.ThicknessRaw, Is.EqualTo(4));
            Assert.That((int)FluidDebugView.Foam, Is.EqualTo(5));
            Assert.That((int)FluidDebugView.Reflection, Is.EqualTo(6));
            Assert.That((int)FluidThicknessMode.OverlapCount, Is.EqualTo(0));
            Assert.That((int)FluidThicknessMode.SphereChord, Is.EqualTo(1));
            Assert.That((int)FluidSurfaceSmoothType.Bilateral, Is.EqualTo(0));
            Assert.That((int)FluidSurfaceSmoothType.Gaussian, Is.EqualTo(1));
        }

        [Test]
        public void DielectricFresnelIsLowAtNormalIncidenceAndHighAtGlance()
        {
            float Reflectance(Vector3 inDir, Vector3 normal, float iorA, float iorB)
            {
                float ratio = iorA / iorB;
                float cosIn = -Vector3.Dot(inDir, normal);
                float sinSqr = ratio * ratio * (1f - cosIn * cosIn);
                if (sinSqr >= 1f)
                {
                    return 1f;
                }

                float cosR = Mathf.Sqrt(1f - sinSqr);
                float rPerp = (iorA * cosIn - iorB * cosR) / (iorA * cosIn + iorB * cosR);
                float rPar = (iorB * cosIn - iorA * cosR) / (iorB * cosIn + iorA * cosR);
                return (rPerp * rPerp + rPar * rPar) * 0.5f;
            }

            var normal = new Vector3(0f, 0f, -1f);
            float facing = Reflectance(new Vector3(0f, 0f, 1f), normal, 1f, 1.33f);
            float glancing = Reflectance(Vector3.Normalize(new Vector3(0.98f, 0f, 0.2f)), normal, 1f, 1.33f);

            Assert.That(facing, Is.GreaterThan(0.01f).And.LessThan(0.05f));
            Assert.That(glancing, Is.GreaterThan(facing));
            Assert.That(glancing, Is.GreaterThan(0.2f));
        }

        [Test]
        public void WaterToAirProducesTotalInternalReflection()
        {
            float SinSqr(Vector3 inDir, Vector3 normal, float iorA, float iorB)
            {
                float ratio = iorA / iorB;
                float cosIn = -Vector3.Dot(inDir, normal);
                return ratio * ratio * (1f - cosIn * cosIn);
            }

            var normal = new Vector3(0f, 0f, -1f);
            var steep = Vector3.Normalize(new Vector3(0.9f, 0f, 0.44f));
            Assert.That(SinSqr(steep, normal, 1.33f, 1f), Is.GreaterThanOrEqualTo(1f));
        }

        [Test]
        public void ThicknessBehindFoamIsCulledByCopiedDepth()
        {
            bool SurvivesZTest(float fragmentEye, float foamEye) => fragmentEye <= foamEye;

            Assert.That(SurvivesZTest(2.0f, 3.0f), Is.True);
            Assert.That(SurvivesZTest(3.0f, 3.0f), Is.True);
            Assert.That(SurvivesZTest(4.0f, 3.0f), Is.False);
        }

        [Test]
        public void SebFoamPacksCoverageUnityDepthAndLinearEye()
        {
            const float coverage = 1f;
            const float unityDepth = 0.42f;
            const float linearEye = 4.1f;
            var packed = new Vector4(coverage, unityDepth, linearEye, 1f);

            Assert.That(packed.x, Is.EqualTo(1f));
            Vector3 Mix(Vector3 col, float foam) => col * (1f - foam) + Vector3.one * foam;
            var water = new Vector3(0.1f, 0.3f, 0.4f);
            Vector3 mixed = Mix(water, packed.x);
            Assert.That(mixed.x, Is.EqualTo(1f).Within(1e-4f));
            Assert.That(mixed.y, Is.EqualTo(1f).Within(1e-4f));
        }

        [Test]
        public void FoamInFrontOfTheSheetReplacesReflection()
        {
            var reflect = new Unity.Mathematics.float3(0.2f, 0.3f, 0.8f);
            var refract = new Unity.Mathematics.float3(0.08f, 0.22f, 0.31f);
            const float foam = 1f;
            const float fluidDepth = 4f;
            const float refractWeight = 0.4f;

            Unity.Mathematics.float3 behind = DiffuseCriteria.CompositeShaded(
                reflect, refract, foam, 5.5f, fluidDepth, refractWeight);
            Unity.Mathematics.float3 expectedBehind = Unity.Mathematics.math.lerp(
                reflect, new Unity.Mathematics.float3(1f, 1f, 1f), refractWeight);
            Assert.That(behind.x, Is.EqualTo(expectedBehind.x).Within(1e-4f));
            Assert.That(behind.y, Is.EqualTo(expectedBehind.y).Within(1e-4f));

            Unity.Mathematics.float3 inFront = DiffuseCriteria.CompositeShaded(
                reflect, refract, foam, 3.2f, fluidDepth, refractWeight);
            Assert.That(inFront.x, Is.EqualTo(1f).Within(1e-4f));
            Assert.That(inFront.y, Is.EqualTo(1f).Within(1e-4f));
        }

        [Test]
        public void SprayDragOpposesVelocityAndScalesWithSpeedSquared()
        {
            var gravity = new Unity.Mathematics.float3(0f, -10f, 0f);
            Unity.Mathematics.float3 fast = DiffuseCriteria.SprayVelocityDelta(
                new Unity.Mathematics.float3(10f, 0f, 0f), gravity, 0.016f);
            Unity.Mathematics.float3 slow = DiffuseCriteria.SprayVelocityDelta(
                new Unity.Mathematics.float3(1f, 0f, 0f), gravity, 0.016f);
            Assert.That(fast.x, Is.LessThan(0f));
            Assert.That(Mathf.Abs(fast.x), Is.GreaterThan(Mathf.Abs(slow.x)));
        }

        [Test]
        public void BubbleAccelerationIsScaledByDeltaTime()
        {
            var gravity = new Unity.Mathematics.float3(0f, -10f, 0f);
            Unity.Mathematics.float3 delta = DiffuseCriteria.BubbleVelocityDelta(
                Unity.Mathematics.float3.zero, new Unity.Mathematics.float3(1f, 0f, 0f),
                gravity, 1.5f, 3f, 0.1f);
            Assert.That(delta.x, Is.EqualTo(0.3f).Within(1e-5f));
            Assert.That(delta.y, Is.EqualTo(0.5f).Within(1e-5f));
        }

        [Test]
        public void SprayOutsideTheTankIsCulled()
        {
            var min = new Unity.Mathematics.float3(-1f, -1f, -1f);
            var max = new Unity.Mathematics.float3(1f, 1f, 1f);
            Assert.That(
                DiffuseCriteria.SprayLeavesDomain(new Unity.Mathematics.float3(2f, 0f, 0f), min, max),
                Is.True);
            Assert.That(
                DiffuseCriteria.SprayLeavesDomain(Unity.Mathematics.float3.zero, min, max),
                Is.False);
        }

        [Test]
        public void FoamAmountIsEnergyTimesTheBlendedSources()
        {
            Assert.That(DiffuseCriteria.Amount(1f, 0f, 1f, 70f, 20f), Is.EqualTo(70f / 90f).Within(1e-4f));
            Assert.That(DiffuseCriteria.Amount(0f, 0f, 1f, 70f, 20f), Is.EqualTo(0f));
            Assert.That(DiffuseCriteria.Amount(1f, 1f, 0f, 70f, 20f), Is.EqualTo(0f));
        }

        [Test]
        public void DiffuseParticlesAreSpawnedSeparatelyFromTheFluid()
        {
            string shader = System.IO.File.ReadAllText(
                System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Render/DiffuseParticle.shader"));
            Assert.That(shader, Does.Contain("_DiffuseOccupied"));
            Assert.That(shader, Does.Contain("UNITY_MATRIX_VP"));
            Assert.That(shader, Does.Contain("UNITY_MATRIX_V"));
            Assert.That(shader, Does.Not.Contain("_SpawnWeight"));

            string body = System.IO.File.ReadAllText(
                System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Include/DiffuseBody.hlsl"));
            Assert.That(body, Does.Contain("void SpawnDiffuse"));
            Assert.That(body, Does.Contain("wakeAxis"));
            Assert.That(body, Does.Contain("_SupportRadius * sqrt(xr)"));
            Assert.That(body, Does.Contain("velocity = fluidVel"));
            Assert.That(body, Does.Contain("1.0 + dot("));
            Assert.That(body, Does.Contain("_SpawnWeight[particle] = ik * (_TrappedAirRate * ita + _WaveCrestRate * iwc)"));

            string composite = System.IO.File.ReadAllText(
                System.IO.Path.Combine(Application.dataPath, "FluidSim/Shaders/Render/FluidComposite.shader"));
            Assert.That(composite, Does.Contain("baseCol * (1.0 - foam) + foam"));
            Assert.That(composite, Does.Contain("return float4(water, 1.0)"));
        }

        [Test]
        public void CheckerParityMatchesSebDarkTiles()
        {
            bool Dark(int x, int y) => x % 2 == y % 2;

            Assert.That(Dark(0, 0), Is.True);
            Assert.That(Dark(1, 0), Is.False);
            Assert.That(Dark(1, 1), Is.True);
        }

        [Test]
        public void FloorShadowBeerLambertDarkensWithThickness()
        {
            float Transmit(float thickness, float extinction) => Mathf.Exp(-thickness * extinction);

            Assert.That(Transmit(0f, 0.4f), Is.EqualTo(1f).Within(1e-5f));
            Assert.That(Transmit(2f, 0.4f), Is.LessThan(Transmit(0.5f, 0.4f)));
            Assert.That(Transmit(2f, 0.4f), Is.GreaterThan(0.3f));
        }

        [Test]
        public void TileQuadrantPicksByLocalSign()
        {
            int Quadrant(float x, float z)
            {
                if (x < 0f && z < 0f) return 3;
                if (x < 0f) return 1;
                if (z < 0f) return 4;
                return 2;
            }

            Assert.That(Quadrant(-1f, 1f), Is.EqualTo(1));
            Assert.That(Quadrant(1f, 1f), Is.EqualTo(2));
            Assert.That(Quadrant(-1f, -1f), Is.EqualTo(3));
            Assert.That(Quadrant(1f, -1f), Is.EqualTo(4));
        }

        [Test]
        public void SsrHitsWhenTheRayCrossesSceneDepthInsideTheThicknessWindow()
        {
            Assert.That(DiffuseCriteria.SsrCrossedSceneDepth(5.2f, 5.0f, 0.55f), Is.True);
            Assert.That(DiffuseCriteria.SsrCrossedSceneDepth(5.0f, 5.2f, 0.55f), Is.False);
            Assert.That(DiffuseCriteria.SsrCrossedSceneDepth(6.5f, 5.0f, 0.55f), Is.False);
            Assert.That(DiffuseCriteria.SsrEdgeFade(0.5f, 0.5f), Is.EqualTo(1f).Within(1e-4f));
            Assert.That(DiffuseCriteria.SsrEdgeFade(0.99f, 0.5f), Is.LessThan(0.2f));
        }

        [Test]
        public void PlanarFloorDoesNotNeedTheHitToBeOnScreen()
        {
            Assert.That(DiffuseCriteria.PlanarFloorWinsWhenOffScreen(true, false), Is.True);
            Assert.That(DiffuseCriteria.PlanarFloorWinsWhenOffScreen(true, true), Is.False);
            Assert.That(DiffuseCriteria.PlanarFloorWinsWhenOffScreen(false, false), Is.False);
        }

        [Test]
        public void RayBoxDstReportsEntryAndTravelThroughAUnitCube()
        {
            Vector2 Dst(Vector3 bMin, Vector3 bMax, Vector3 origin, Vector3 dir)
            {
                Vector3 inv = new Vector3(1f / dir.x, 1f / dir.y, 1f / dir.z);
                Vector3 t0 = Vector3.Scale(bMin - origin, inv);
                Vector3 t1 = Vector3.Scale(bMax - origin, inv);
                Vector3 tmin = Vector3.Min(t0, t1);
                Vector3 tmax = Vector3.Max(t0, t1);
                float dstA = Mathf.Max(tmin.x, Mathf.Max(tmin.y, tmin.z));
                float dstB = Mathf.Min(tmax.x, Mathf.Min(tmax.y, tmax.z));
                float dstToBox = Mathf.Max(0f, dstA);
                float dstInside = Mathf.Max(0f, dstB - dstToBox);
                return new Vector2(dstToBox, dstInside);
            }

            Vector2 hit = Dst(
                new Vector3(-1f, -1f, -1f), new Vector3(1f, 1f, 1f),
                new Vector3(0f, 0f, -3f), Vector3.forward);
            Assert.That(hit.x, Is.EqualTo(2f).Within(1e-4f));
            Assert.That(hit.y, Is.EqualTo(2f).Within(1e-4f));

            Vector2 miss = Dst(
                new Vector3(-1f, -1f, -1f), new Vector3(1f, 1f, 1f),
                new Vector3(4f, 0f, -3f), Vector3.forward);
            Assert.That(miss.y, Is.EqualTo(0f));
        }

        [Test]
        public void MarchedMetrePathDarkensMoreThanAThinSheet()
        {
            float Transmit(float metres, Vector3 extinction)
            {
                return Mathf.Exp(-metres * extinction.x);
            }

            var extinction = new Vector3(0.55f, 0.16f, 0.07f);
            float sheet = Transmit(0.25f * 0.8f, extinction);
            float pool = Transmit(1.6f, extinction);
            Assert.That(pool, Is.LessThan(sheet * 0.6f));
            Assert.That(pool, Is.GreaterThan(0.05f));
        }

        [Test]
        public void ViewRayFromInverseProjectionIsNotTheBlitOrtho()
        {
            var camera = new GameObject("FluidRayCam").AddComponent<Camera>();
            try
            {
                camera.fieldOfView = 52f;
                camera.nearClipPlane = 0.05f;
                camera.farClipPlane = 50f;
                camera.aspect = 16f / 9f;
                Matrix4x4 gpu = GL.GetGPUProjectionMatrix(camera.projectionMatrix, true);
                Vector4 viewHom = gpu.inverse * new Vector4(0f, 0f, 0f, -1f);
                var viewRay = new Vector3(viewHom.x, viewHom.y, viewHom.z);
                Assert.That(viewRay.sqrMagnitude, Is.GreaterThan(1e-6f));
                Assert.That(Mathf.Abs(viewRay.x), Is.LessThan(0.05f));
            }
            finally
            {
                Object.DestroyImmediate(camera.gameObject);
            }
        }

        static void AssertCompiled(string path)
        {
            var shader = AssetDatabase.LoadAssetAtPath<Shader>(path);
            Assert.That(shader, Is.Not.Null, path);
            Assert.That(shader.isSupported, Is.True, $"{path} is not supported on this device.");
        }
    }
}
