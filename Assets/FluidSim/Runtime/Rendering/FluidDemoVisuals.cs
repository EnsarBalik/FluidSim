using FluidSim.Core;
using FluidSim.Solver;
using UnityEngine;

namespace FluidSim.Rendering
{
    /// <summary>
    /// Game-view stand-ins for the tank, tiled floor and the demo rigid box.
    /// Physics already treats those as Akinci samples; without meshes the water
    /// floats in empty space and Green's scene grab / depth test have nothing
    /// to catch. The floor is SebLague-style checker tiles so refraction reads.
    /// </summary>
    [AddComponentMenu("FluidSim/Fluid Demo Visuals")]
    [ExecuteAlways]
    [DefaultExecutionOrder(50)]
    public sealed class FluidDemoVisuals : MonoBehaviour
    {
        const string SolidShaderName = "Hidden/FluidSim/DemoSolid";
        const string FloorShaderName = "Hidden/FluidSim/DemoFloor";
        const string GlassShaderName = "Hidden/FluidSim/DemoGlass";

        [SerializeField]
        FluidSimulation simulation;

        [SerializeField]
        Shader solidShader;

        [SerializeField]
        Shader floorShader;

        [SerializeField]
        Shader glassShader;

        [SerializeField]
        bool drawTank = true;

        [SerializeField]
        bool drawFloor = true;

        [SerializeField]
        bool drawRigidBox = true;

        [SerializeField]
        Color glassColor = new Color(0.62f, 0.8f, 0.9f, 0.11f);

        [SerializeField]
        Color rigidColor = new Color(0.92f, 0.45f, 0.16f, 1f);

        [Header("Floor tiles")]
        [SerializeField]
        Color tileCol1 = new Color(0.82f, 0.78f, 0.72f, 1f);

        [SerializeField]
        Color tileCol2 = new Color(0.76f, 0.80f, 0.84f, 1f);

        [SerializeField]
        Color tileCol3 = new Color(0.80f, 0.74f, 0.70f, 1f);

        [SerializeField]
        Color tileCol4 = new Color(0.72f, 0.76f, 0.78f, 1f);

        [SerializeField]
        Vector3 tileColVariation = new Vector3(0.02f, 0.05f, 0.06f);

        [SerializeField, Min(0.1f)]
        float tileScale = 1.5f;

        [SerializeField]
        float tileDarkOffset = -0.18f;

        [SerializeField, Min(4f)]
        float floorExtent = 24f;

        Mesh cubeMesh;
        Material glassMaterial;
        Material floorMaterial;
        Material rigidMaterial;

        static readonly int TileCol1Id = Shader.PropertyToID("_TileCol1");
        static readonly int TileCol2Id = Shader.PropertyToID("_TileCol2");
        static readonly int TileCol3Id = Shader.PropertyToID("_TileCol3");
        static readonly int TileCol4Id = Shader.PropertyToID("_TileCol4");
        static readonly int TileColVariationId = Shader.PropertyToID("_TileColVariation");
        static readonly int TileScaleId = Shader.PropertyToID("_TileScale");
        static readonly int TileDarkOffsetId = Shader.PropertyToID("_TileDarkOffset");
        static readonly int TileOriginId = Shader.PropertyToID("_TileOrigin");

        void OnEnable()
        {
            if (simulation == null)
            {
                simulation = GetComponent<FluidSimulation>();
            }

            cubeMesh = Resources.GetBuiltinResource<Mesh>("Cube.fbx");
            if (cubeMesh == null)
            {
                var temp = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cubeMesh = temp.GetComponent<MeshFilter>().sharedMesh;
                DestroyImmediate(temp);
            }

            if (!CreateMaterials())
            {
                enabled = false;
            }
        }

        void OnDisable()
        {
            DestroyMaterial(ref glassMaterial);
            DestroyMaterial(ref floorMaterial);
            DestroyMaterial(ref rigidMaterial);
        }

        void LateUpdate()
        {
            if (simulation == null || simulation.Parameters == null || cubeMesh == null)
            {
                return;
            }

            Bounds domain = simulation.Parameters.Domain;
            if (drawTank && glassMaterial != null)
            {
                glassMaterial.SetColor("_Color", glassColor);
                DrawTankShell(domain, 0.03f);
            }

            if (drawFloor && floorMaterial != null)
            {
                GetFloorBounds(domain, out Vector3 center, out Vector3 size);
                ApplyTiles(floorMaterial);
                DrawSolid(center, Quaternion.identity, size, floorMaterial);
            }

            if (!drawRigidBox || !simulation.SpawnDemoRigidBox || rigidMaterial == null)
            {
                return;
            }

            Vector3 rigidCenter = simulation.DemoRigidBoxCenter;
            Quaternion rigidRotation = Quaternion.identity;
            Vector3 rigidSize = simulation.DemoRigidBoxSize;
            if (Application.isPlaying && simulation.RigidBody != null)
            {
                rigidCenter = (Vector3)simulation.RigidBody.Position;
                rigidRotation = (Quaternion)simulation.RigidBody.Rotation;
                rigidSize = (Vector3)simulation.RigidBody.Size;
            }
            else if (simulation.Parameters.Dimension == SimulationDimension.Two)
            {
                rigidCenter.z = domain.center.z;
                rigidSize.z = 0.02f;
            }

            rigidMaterial.SetColor("_Color",
                simulation.IsDraggingRigid
                    ? Color.Lerp(rigidColor, Color.white, 0.35f)
                    : rigidColor);
            DrawSolid(rigidCenter, rigidRotation, rigidSize, rigidMaterial);
        }

        public void GetFloorBounds(Bounds domain, out Vector3 center, out Vector3 size)
        {
            DefaultFloorBounds(domain, floorExtent, out center, out size);
        }

        public static void DefaultFloorBounds(
            Bounds domain, float extent, out Vector3 center, out Vector3 size)
        {
            size = domain.size;
            center = domain.center;
            center.y = domain.min.y - 0.04f;
            size.y = 0.08f;
            size.x = Mathf.Max(size.x + 0.4f, extent);
            size.z = Mathf.Max(size.z + 0.4f, extent);
        }

        public void ApplyTiles(Material material)
        {
            if (material == null)
            {
                return;
            }

            BindFloor(material, simulation != null && simulation.Parameters != null
                ? simulation.Parameters.Domain.center
                : Vector3.zero);
        }

        void BindFloor(Material material, Vector3 origin)
        {
            material.SetColor(TileCol1Id, tileCol1);
            material.SetColor(TileCol2Id, tileCol2);
            material.SetColor(TileCol3Id, tileCol3);
            material.SetColor(TileCol4Id, tileCol4);
            material.SetVector(TileColVariationId, tileColVariation);
            material.SetFloat(TileScaleId, tileScale);
            material.SetFloat(TileDarkOffsetId, tileDarkOffset);
            material.SetVector(TileOriginId, origin);
        }

        void DrawTankShell(Bounds domain, float thickness)
        {
            Vector3 center = domain.center;
            Vector3 size = domain.size;
            float t = Mathf.Max(thickness, 0.01f);
            DrawBox(center + new Vector3(size.x * 0.5f - t * 0.5f, 0f, 0f), new Vector3(t, size.y, size.z), glassMaterial);
            DrawBox(center - new Vector3(size.x * 0.5f - t * 0.5f, 0f, 0f), new Vector3(t, size.y, size.z), glassMaterial);
            DrawBox(center + new Vector3(0f, size.y * 0.5f - t * 0.5f, 0f), new Vector3(size.x, t, size.z), glassMaterial);
            DrawBox(center - new Vector3(0f, size.y * 0.5f - t * 0.5f, 0f), new Vector3(size.x, t, size.z), glassMaterial);
            DrawBox(center + new Vector3(0f, 0f, size.z * 0.5f - t * 0.5f), new Vector3(size.x, size.y, t), glassMaterial);
            DrawBox(center - new Vector3(0f, 0f, size.z * 0.5f - t * 0.5f), new Vector3(size.x, size.y, t), glassMaterial);
        }

        void DrawSolid(Vector3 center, Quaternion rotation, Vector3 size, Material material)
        {
            Graphics.DrawMesh(
                cubeMesh, Matrix4x4.TRS(center, rotation, size),
                material, gameObject.layer, null, 0, null,
                UnityEngine.Rendering.ShadowCastingMode.On, true);
        }

        void DrawBox(Vector3 center, Vector3 size, Material material)
        {
            Graphics.DrawMesh(
                cubeMesh, Matrix4x4.TRS(center, Quaternion.identity, size),
                material, gameObject.layer, null, 0, null, false, false, false);
        }

        bool CreateMaterials()
        {
            Shader solid = solidShader != null ? solidShader : Shader.Find(SolidShaderName);
            Shader floor = floorShader != null ? floorShader : Shader.Find(FloorShaderName);
            Shader glass = glassShader != null ? glassShader : Shader.Find(GlassShaderName);
            if (solid == null || glass == null)
            {
                Debug.LogError(
                    "Could not find the demo visual shaders. Recreate from " +
                    "GameObject > FluidSim > Fluid Simulation.", this);
                return false;
            }

            if (floor == null)
            {
                floor = solid;
            }

            glassMaterial = new Material(glass) { hideFlags = HideFlags.HideAndDontSave };
            floorMaterial = new Material(floor) { hideFlags = HideFlags.HideAndDontSave };
            rigidMaterial = new Material(solid) { hideFlags = HideFlags.HideAndDontSave };
            return true;
        }

        static void DestroyMaterial(ref Material material)
        {
            if (material != null)
            {
                DestroyImmediate(material);
                material = null;
            }
        }
    }
}
