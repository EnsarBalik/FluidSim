using System;
using FluidSim.Core;
using FluidSim.Neighbors;
using Unity.Mathematics;
using UnityEngine;

namespace FluidSim.Solver
{
    /// <summary>
    /// Akinci 2012 dynamic boundary: surface samples whose world positions follow a
    /// rigid pose. Particle masses are SPH density weights; inertial mass is the
    /// solid volume times the authored density and is what the two-way coupling
    /// integrates.
    /// </summary>
    public sealed class SphRigidBody : IDisposable
    {
        readonly ComputeShader gridShader;
        readonly float3[] localSamples;
        readonly float3[] worldPositions3D;
        readonly float2[] worldPositions2D;
        readonly float3[] worldVelocities3D;
        readonly float2[] worldVelocities2D;
        readonly float3[] cornerOffsets;

        public ParticleSet Particles { get; }

        public NeighborGrid Grid { get; private set; }

        public ComputeBuffer Masses { get; private set; }

        public ComputeBuffer UnsortedMasses { get; private set; }

        public ComputeBuffer Forces { get; private set; }

        public ComputeBuffer ReducedForceAndTorque { get; private set; }

        public bool GridReady { get; set; }

        public bool MassesCaptured { get; set; }

        public float3 Size { get; }

        public float Mass { get; }

        public float3 InertiaDiagonal { get; }

        public bool IsKinematic { get; set; }

        public float3 Position;

        public quaternion Rotation = quaternion.identity;

        public float3 LinearVelocity;

        public float3 AngularVelocity;

        public int Count => Particles != null ? Particles.Count : 0;

        public SphRigidBody(
            ComputeShader gridShader,
            FluidParameters parameters,
            float3 size,
            float3 position,
            float density,
            bool kinematic = false)
        {
            if (gridShader == null)
            {
                throw new ArgumentNullException(nameof(gridShader));
            }

            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }

            this.gridShader = gridShader;
            Size = math.max(size, new float3(parameters.ParticleSize));
            Position = position;
            IsKinematic = kinematic;

            localSamples = RigidSurfaceSampler.SampleBox(Size, parameters.ParticleSize, parameters.Dimension);
            if (localSamples.Length == 0)
            {
                throw new InvalidOperationException("The rigid box produced no surface samples.");
            }

            float volume = parameters.Dimension == SimulationDimension.Two
                ? Size.x * Size.y
                : Size.x * Size.y * Size.z;
            Mass = math.max(density, 1e-3f) * volume;

            // Solid box inertia about the centre of mass. In 2D only Izz is used.
            float3 s2 = Size * Size;
            InertiaDiagonal = Mass / 12f * new float3(s2.y + s2.z, s2.x + s2.z, s2.x + s2.y);

            Particles = new ParticleSet(localSamples.Length, parameters.DimensionCount);
            worldPositions3D = new float3[localSamples.Length];
            worldVelocities3D = new float3[localSamples.Length];
            worldPositions2D = new float2[localSamples.Length];
            worldVelocities2D = new float2[localSamples.Length];
            cornerOffsets = new float3[8];
            int corner = 0;
            float3 half = Size * 0.5f;
            for (int z = -1; z <= 1; z += 2)
            {
                for (int y = -1; y <= 1; y += 2)
                {
                    for (int x = -1; x <= 1; x += 2)
                    {
                        cornerOffsets[corner++] = new float3(x, y, z) * half;
                    }
                }
            }

            ApplyPose();
            if (Particles.Dimensions == 3)
            {
                Particles.Fill(worldPositions3D);
            }
            else
            {
                Particles.Fill(worldPositions2D);
            }

            ApplyPose();
            Grid = new NeighborGrid(gridShader, parameters, localSamples.Length);
            Grid.Build(Particles, parameters);

            Masses = new ComputeBuffer(localSamples.Length, sizeof(float));
            UnsortedMasses = new ComputeBuffer(localSamples.Length, sizeof(float));
            Forces = new ComputeBuffer(localSamples.Length, sizeof(float) * 3);
            ReducedForceAndTorque = new ComputeBuffer(2, sizeof(float) * 3);
            GridReady = true;
            ClearForces();
        }

        public void SetPose(float3 position, quaternion rotation)
        {
            Position = position;
            Rotation = rotation;
            LinearVelocity = float3.zero;
            AngularVelocity = float3.zero;
            GridReady = false;
        }

        /// <summary>
        /// User drag: the solid follows the pointer and sample velocities match the
        /// motion so the fluid still feels the kick. Gravity and fluid forces stay
        /// off for the duration of the grab.
        /// </summary>
        public void DragTo(float3 worldPosition, float deltaTime, Bounds domain, float maxSpeed = 12f)
        {
            if (Particles.Dimensions == 2)
            {
                worldPosition.z = domain.center.z;
            }

            float dt = math.max(deltaTime, 1e-6f);
            LinearVelocity = (worldPosition - Position) / dt;
            float speed = math.length(LinearVelocity);
            if (speed > maxSpeed)
            {
                LinearVelocity *= maxSpeed / speed;
            }

            AngularVelocity = float3.zero;
            Position = worldPosition;
            CollideDomain(domain);
            GridReady = false;
        }

        public bool TryRaycast(Ray ray, out float distance)
        {
            float3 pickSize = Size * 1.08f;
            if (Particles.Dimensions == 2)
            {
                pickSize.z = math.max(pickSize.z, 0.45f);
            }

            return RaycastOrientedBox(ray, Position, Rotation, pickSize, out distance);
        }

        public static bool RaycastOrientedBox(
            Ray ray, float3 center, quaternion rotation, float3 size, out float distance)
        {
            distance = 0f;
            float3 half = size * 0.5f;
            if (half.x <= 0f || half.y <= 0f || half.z <= 0f)
            {
                return false;
            }

            float3x3 worldFromLocal = new float3x3(rotation);
            float3x3 localFromWorld = math.transpose(worldFromLocal);
            float3 origin = math.mul(localFromWorld, (float3)ray.origin - center);
            float3 direction = math.mul(localFromWorld, (float3)ray.direction);

            float tMin = float.NegativeInfinity;
            float tMax = float.PositiveInfinity;
            for (int axis = 0; axis < 3; axis++)
            {
                float o = origin[axis];
                float d = direction[axis];
                float h = half[axis];
                if (math.abs(d) < 1e-8f)
                {
                    if (o < -h || o > h)
                    {
                        return false;
                    }

                    continue;
                }

                float inv = 1f / d;
                float t0 = (-h - o) * inv;
                float t1 = (h - o) * inv;
                if (t0 > t1)
                {
                    (t0, t1) = (t1, t0);
                }

                tMin = math.max(tMin, t0);
                tMax = math.min(tMax, t1);
                if (tMin > tMax)
                {
                    return false;
                }
            }

            if (tMax < 0f)
            {
                return false;
            }

            distance = tMin >= 0f ? tMin : 0f;
            return true;
        }

        /// <summary>
        /// Writes world-space sample positions and rigid velocities
        /// <c>V + ω × (x − X)</c> into the unsorted particle buffers.
        /// </summary>
        public void ApplyPose()
        {
            for (int i = 0; i < localSamples.Length; i++)
            {
                float3 world = Position + math.mul(Rotation, localSamples[i]);
                float3 velocity = LinearVelocity + math.cross(AngularVelocity, world - Position);
                worldPositions3D[i] = world;
                worldVelocities3D[i] = velocity;
                worldPositions2D[i] = world.xy;
                worldVelocities2D[i] = velocity.xy;
            }

            if (Particles.Dimensions == 3)
            {
                Particles.Positions.SetData(worldPositions3D);
                Particles.Velocities.SetData(worldVelocities3D);
            }
            else
            {
                Particles.Positions.SetData(worldPositions2D);
                Particles.Velocities.SetData(worldVelocities2D);
            }
        }

        public void RebuildGrid(FluidParameters parameters)
        {
            ApplyPose();
            Grid.Build(Particles, parameters);
            GridReady = true;
        }

        public void ReallocateGrid(FluidParameters parameters)
        {
            Grid?.Dispose();
            Grid = new NeighborGrid(gridShader, parameters, localSamples.Length);
            GridReady = false;
            MassesCaptured = false;
        }

        public void ClearForces()
        {
            Forces.SetData(new float3[localSamples.Length]);
        }

        public void Integrate(float3[] particleForces, float deltaTime, float3 gravity, Bounds domain)
        {
            if (IsKinematic || particleForces == null || particleForces.Length != localSamples.Length)
            {
                return;
            }

            float3 force = float3.zero;
            float3 torque = float3.zero;
            float3[] samples = Particles.Dimensions == 3
                ? Particles.ReadSortedPositions3D()
                : Promote(Particles.ReadSortedPositions2D());

            for (int i = 0; i < particleForces.Length; i++)
            {
                force += particleForces[i];
                torque += math.cross(samples[i] - Position, particleForces[i]);
            }

            Integrate(force, torque, deltaTime, gravity, domain);
        }

        public void Integrate(float3 fluidForce, float3 torque, float deltaTime, float3 gravity, Bounds domain)
        {
            if (IsKinematic)
            {
                return;
            }

            LinearVelocity += deltaTime * ((fluidForce + Mass * gravity) / math.max(Mass, 1e-6f));
            AngularVelocity += deltaTime * ApplyInverseInertia(torque);
            Position += deltaTime * LinearVelocity;
            Rotation = IntegrateRotation(Rotation, AngularVelocity, deltaTime);
            CollideDomain(domain);
            GridReady = false;
        }

        public void Dispose()
        {
            Grid?.Dispose();
            Particles?.Dispose();
            Masses?.Release();
            UnsortedMasses?.Release();
            Forces?.Release();
            ReducedForceAndTorque?.Release();
            Masses = null;
            UnsortedMasses = null;
            Forces = null;
            ReducedForceAndTorque = null;
            GridReady = false;
            MassesCaptured = false;
        }

        float3 ApplyInverseInertia(float3 torque)
        {
            if (Particles.Dimensions == 2)
            {
                return new float3(0f, 0f, torque.z / math.max(InertiaDiagonal.z, 1e-8f));
            }

            float3x3 rotation = new float3x3(Rotation);
            float3x3 inverseLocal = float3x3.zero;
            inverseLocal.c0.x = 1f / math.max(InertiaDiagonal.x, 1e-8f);
            inverseLocal.c1.y = 1f / math.max(InertiaDiagonal.y, 1e-8f);
            inverseLocal.c2.z = 1f / math.max(InertiaDiagonal.z, 1e-8f);
            float3x3 inverseWorld = math.mul(rotation, math.mul(inverseLocal, math.transpose(rotation)));
            return math.mul(inverseWorld, torque);
        }

        void CollideDomain(Bounds domain)
        {
            float3 minCorner = new float3(float.PositiveInfinity);
            float3 maxCorner = new float3(float.NegativeInfinity);
            int corners = Particles.Dimensions == 2 ? 4 : 8;
            for (int i = 0; i < corners; i++)
            {
                float3 world = Position + math.mul(Rotation, cornerOffsets[i]);
                if (Particles.Dimensions == 2)
                {
                    world.z = domain.center.z;
                }

                minCorner = math.min(minCorner, world);
                maxCorner = math.max(maxCorner, world);
            }

            float3 push = float3.zero;
            float3 minBound = (float3)domain.min;
            float3 maxBound = (float3)domain.max;
            if (Particles.Dimensions == 2)
            {
                minBound.z = domain.center.z;
                maxBound.z = domain.center.z;
            }

            if (minCorner.x < minBound.x)
            {
                push.x += minBound.x - minCorner.x;
            }

            if (maxCorner.x > maxBound.x)
            {
                push.x += maxBound.x - maxCorner.x;
            }

            if (minCorner.y < minBound.y)
            {
                push.y += minBound.y - minCorner.y;
            }

            if (maxCorner.y > maxBound.y)
            {
                push.y += maxBound.y - maxCorner.y;
            }

            if (Particles.Dimensions == 3)
            {
                if (minCorner.z < minBound.z)
                {
                    push.z += minBound.z - minCorner.z;
                }

                if (maxCorner.z > maxBound.z)
                {
                    push.z += maxBound.z - maxCorner.z;
                }
            }

            if (math.lengthsq(push) <= 0f)
            {
                return;
            }

            Position += push;
            if (push.x != 0f && math.sign(LinearVelocity.x) != math.sign(push.x))
            {
                LinearVelocity.x = 0f;
            }

            if (push.y != 0f && math.sign(LinearVelocity.y) != math.sign(push.y))
            {
                LinearVelocity.y = 0f;
            }

            if (push.z != 0f && math.sign(LinearVelocity.z) != math.sign(push.z))
            {
                LinearVelocity.z = 0f;
            }
        }

        static quaternion IntegrateRotation(quaternion rotation, float3 angularVelocity, float deltaTime)
        {
            float speed = math.length(angularVelocity);
            if (speed < 1e-8f)
            {
                return math.normalize(rotation);
            }

            quaternion step = quaternion.AxisAngle(angularVelocity / speed, speed * deltaTime);
            return math.normalize(math.mul(step, rotation));
        }

        static float3[] Promote(float2[] flat)
        {
            var promoted = new float3[flat.Length];
            for (int i = 0; i < flat.Length; i++)
            {
                promoted[i] = new float3(flat[i], 0f);
            }

            return promoted;
        }
    }
}
