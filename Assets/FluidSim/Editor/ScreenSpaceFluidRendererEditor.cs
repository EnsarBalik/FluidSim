using FluidSim.Rendering;
using UnityEditor;
using UnityEngine;

namespace FluidSim.Editor
{
    [CustomEditor(typeof(ScreenSpaceFluidRenderer))]
    public sealed class ScreenSpaceFluidRendererEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "Open the Game view (not the Scene view). Scene view is diagnostic " +
                "discs. Game view ray-marches SebLague's Raymarching.shader: particles " +
                "are voxelised into a 3D SPH density map, then FindNextSurface walks the " +
                "isosurface (density − Density Offset) so a cluster is one volume, not " +
                "separate spheres. Each bounce traces the more interesting of reflect vs " +
                "refract (optical depth × Fresnel). Reflections sample the tiled floor " +
                "at the world hit (planar, works off-screen) and march the scene depth " +
                "for the box / tank (SSR). Debug View = Reflection paints SSR red, " +
                "floor green, cube yellow, sky blue. Lower Density Offset if individual " +
                "particles still show; raise it to tighten the surface toward rest density. " +
                "Foam is extra spawned BillboardFoam discs, drawn first so thickness does " +
                "not sit in front of spray.",
                MessageType.Info);
        }
    }
}
