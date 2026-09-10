using UnityEngine;
using UnityEngine.InputSystem;

namespace FluidSim
{
    /// <summary>
    /// Play Mode fly cam for the Game view. WASD strafes, Q/E go up and down,
    /// Shift boosts, right mouse looks. Left click stays free for the rigid box.
    /// </summary>
    [AddComponentMenu("FluidSim/Free Fly Camera")]
    [DisallowMultipleComponent]
    public sealed class FreeFlyCamera : MonoBehaviour
    {
        [SerializeField, Min(0.1f)]
        float moveSpeed = 6f;

        [SerializeField, Min(1f)]
        float boostMultiplier = 3f;

        [SerializeField, Min(0.01f)]
        float lookSensitivity = 0.12f;

        [SerializeField]
        Vector2 pitchLimits = new Vector2(-89f, 89f);

        float yaw;
        float pitch;
        bool looking;
        CursorLockMode previousLock;
        bool previousCursorVisible;

        void OnEnable()
        {
            ReadOrientation();
        }

        void OnDisable()
        {
            StopLooking();
        }

        void Update()
        {
            if (!Application.isPlaying)
            {
                return;
            }

            Mouse mouse = Mouse.current;
            Keyboard keyboard = Keyboard.current;
            if (mouse == null || keyboard == null)
            {
                return;
            }

            bool overGame = PointerIsOverGameView();
            if (mouse.rightButton.wasPressedThisFrame && overGame && !mouse.leftButton.isPressed)
            {
                StartLooking();
            }

            if (looking && !mouse.rightButton.isPressed)
            {
                StopLooking();
            }

            if (looking)
            {
                Vector2 delta = mouse.delta.ReadValue();
                yaw += delta.x * lookSensitivity;
                pitch -= delta.y * lookSensitivity;
                pitch = Mathf.Clamp(pitch, pitchLimits.x, pitchLimits.y);
                transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
            }

            if (!looking && !overGame)
            {
                return;
            }

            Vector3 local = Vector3.zero;
            if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed)
            {
                local.z += 1f;
            }

            if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed)
            {
                local.z -= 1f;
            }

            if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed)
            {
                local.x -= 1f;
            }

            if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed)
            {
                local.x += 1f;
            }

            if (keyboard.eKey.isPressed || keyboard.spaceKey.isPressed)
            {
                local.y += 1f;
            }

            if (keyboard.qKey.isPressed)
            {
                local.y -= 1f;
            }

            if (local.sqrMagnitude < 1e-6f)
            {
                return;
            }

            float speed = moveSpeed;
            if (keyboard.leftShiftKey.isPressed || keyboard.rightShiftKey.isPressed)
            {
                speed *= boostMultiplier;
            }

            transform.position += transform.rotation * local.normalized * (speed * Time.deltaTime);
        }

        void ReadOrientation()
        {
            Vector3 euler = transform.eulerAngles;
            yaw = euler.y;
            pitch = euler.x > 180f ? euler.x - 360f : euler.x;
            pitch = Mathf.Clamp(pitch, pitchLimits.x, pitchLimits.y);
        }

        void StartLooking()
        {
            if (looking)
            {
                return;
            }

            looking = true;
            previousLock = Cursor.lockState;
            previousCursorVisible = Cursor.visible;
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        void StopLooking()
        {
            if (!looking)
            {
                return;
            }

            looking = false;
            Cursor.lockState = previousLock;
            Cursor.visible = previousCursorVisible;
        }

        static bool PointerIsOverGameView()
        {
#if UNITY_EDITOR
            System.Type windowType = System.Type.GetType("UnityEditor.EditorWindow,UnityEditor");
            object hovered = windowType?
                .GetProperty("mouseOverWindow",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)?
                .GetValue(null, null);
            return hovered != null && hovered.GetType().Name == "GameView";
#else
            return true;
#endif
        }
    }
}
