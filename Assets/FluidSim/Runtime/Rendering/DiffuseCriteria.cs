using Unity.Mathematics;

namespace FluidSim.Rendering
{
    /// <summary>
    /// Ihmsen et al. 2012, Eq. (1): map a raw potential into [0, 1] between two thresholds.
    /// Kept in C# so the spawn tests cannot drift from the compute shader.
    /// </summary>
    public static class DiffuseCriteria
    {
        public const int SprayNeighborLimit3D = 6;
        public const int BubbleNeighborLimit3D = 20;
        public const int SprayNeighborLimit2D = 4;
        public const int BubbleNeighborLimit2D = 12;

        public static float Phi(float value, float minThreshold, float maxThreshold)
        {
            float range = math.max(maxThreshold - minThreshold, 1e-5f);
            return math.saturate((math.min(value, maxThreshold) - math.min(value, minThreshold)) / range);
        }

        public static int Classify(int fluidNeighbors, int dimensions)
        {
            int sprayLimit = dimensions == 2 ? SprayNeighborLimit2D : SprayNeighborLimit3D;
            int bubbleLimit = dimensions == 2 ? BubbleNeighborLimit2D : BubbleNeighborLimit3D;
            if (fluidNeighbors < sprayLimit)
            {
                return 0;
            }

            return fluidNeighbors > bubbleLimit ? 2 : 1;
        }

        public const float SprayDrag = 0.04f;
        public const float CollisionDamping = 0.1f;
        public const float DefaultFoamLifeMin = 5f;
        public const float DefaultFoamLifeMax = 15f;
        public const float DefaultBubbleFluidAccel = 3f;
        public const float WaveCrestDot = 0.6f;

        /// <summary>
        /// Pair term for trapped air. Ihmsen eq. (2) uses 1 − v̂·x̂ (impacts /
        /// approaching). We use 1 + v̂·x̂ so a moving solid aerates the wake
        /// instead of the bow.
        /// </summary>
        public static float TrappedAirPair(float3 relativeVelocity, float3 offset)
        {
            float relSpeed = math.length(relativeVelocity);
            float dist = math.length(offset);
            if (relSpeed < 1e-5f || dist < 1e-5f)
            {
                return 0f;
            }

            return relSpeed * (1f + math.dot(relativeVelocity / relSpeed, offset / dist));
        }

        public static bool WaveCrestVelocityGate(float3 velocity, float3 normal)
        {
            float speed = math.length(velocity);
            float nLen = math.length(normal);
            if (speed < 1e-5f || nLen < 1e-5f)
            {
                return false;
            }

            return math.dot(velocity / speed, normal / nLen) <= -WaveCrestDot;
        }

        public static float3 SprayVelocityDelta(float3 velocity, float3 gravity, float deltaTime)
        {
            float sqrSpeed = math.lengthsq(velocity);
            float3 drag = sqrSpeed > 1e-8f
                ? -math.normalize(velocity) * sqrSpeed * SprayDrag
                : float3.zero;
            return (gravity + drag) * deltaTime;
        }

        public static float3 BubbleVelocityDelta(
            float3 velocity, float3 fluidVelocity, float3 gravity, float buoyancy, float fluidAccel, float deltaTime)
        {
            float3 accelBuoyancy = gravity * (1f - buoyancy);
            float3 accelFluid = (fluidVelocity - velocity) * fluidAccel;
            return (accelBuoyancy + accelFluid) * deltaTime;
        }

        public static bool SprayLeavesDomain(float3 position, float3 domainMin, float3 domainMax)
        {
            return math.any(position < domainMin) || math.any(position > domainMax);
        }

        public static float Amount(float trappedAir, float waveCrest, float energy, float trappedAirRate, float waveCrestRate)
        {
            float mixed = (trappedAirRate * trappedAir + waveCrestRate * waveCrest) /
                          math.max(trappedAirRate + waveCrestRate, 1e-5f);
            return energy * math.saturate(mixed);
        }

        public static float3 MixFoam(float3 baseCol, float foam)
        {
            return baseCol * (1f - foam) + foam;
        }

        public static float3 CompositeShaded(
            float3 reflectCol, float3 refractCol, float foam, float foamDepth, float fluidDepth, float refractWeight)
        {
            float3 refract = MixFoam(refractCol, foam);
            float3 reflect = foamDepth < fluidDepth ? MixFoam(reflectCol, foam) : reflectCol;
            return math.lerp(reflect, refract, refractWeight);
        }

        public static float BackgroundBlurWeight(float thickness, float thicknessScale, float blurAmount)
        {
            return math.saturate(thickness * thicknessScale * blurAmount);
        }

        public static float3 MixSceneThroughWater(float3 sharp, float3 blurred, float blurWeight)
        {
            return math.lerp(sharp, blurred, math.saturate(blurWeight));
        }

        public static float3 LiftTransmission(float3 transmission, float reveal)
        {
            float lift = math.saturate(reveal);
            return math.max(transmission, new float3(lift, lift, lift));
        }

        public static bool PreferRefractPath(
            float densityRefract, float refractWeight, float densityReflect, float reflectWeight)
        {
            return densityRefract * refractWeight > densityReflect * reflectWeight;
        }

        public static float SampleDensity(float density, float densityOffset)
        {
            return density - densityOffset;
        }

        public static bool IsInsideFluid(float density, float densityOffset)
        {
            return SampleDensity(density, densityOffset) > 0f;
        }

        public static float MarchDensitySample(float density, float densityMultiplier, float stepSize)
        {
            return math.max(density, 0f) * math.max(densityMultiplier, 0f) * math.max(stepSize, 0f);
        }

        public static float VolumeOpticalDepth(float marched, float sheet)
        {
            return math.max(marched, 0f) + math.max(sheet, 0f);
        }

        public static float3 Absorb(float opticalDepth, float3 absorptionCoefficient)
        {
            return math.exp(-math.max(opticalDepth, 0f) * absorptionCoefficient);
        }

        public static float3 MixInterior(float3 behind, float3 bent, float interiorMix)
        {
            return math.lerp(bent, behind, math.saturate(interiorMix));
        }

        public static bool SsrCrossedSceneDepth(float rayEye, float sceneEye, float thickness)
        {
            float delta = rayEye - sceneEye;
            return delta > 0.02f && delta < math.max(thickness, 0.05f);
        }

        public static float SsrEdgeFade(float u, float v)
        {
            float2 e = math.abs(new float2(u, v) * 2f - 1f);
            return 1f - math.saturate((math.max(e.x, e.y) - 0.84f) / 0.14f);
        }

        public static bool PlanarFloorWinsWhenOffScreen(bool floorHit, bool hitOnScreen)
        {
            return floorHit && !hitOnScreen;
        }
    }
}
