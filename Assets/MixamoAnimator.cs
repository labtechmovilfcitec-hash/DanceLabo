using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// REQUIRES: com.unity.nuget.newtonsoft-json package in Unity (via Package Manager)
using Newtonsoft.Json.Linq;

public class MixamoAnimator : MonoBehaviour
{
    public UDPClient udpClient;

    [Header("Mixamo Bones")]
    public Transform leftUpLeg;
    public Transform leftLeg;
    public Transform leftFoot;
    public Transform rightUpLeg;
    public Transform rightLeg;
    public Transform rightFoot;

    public Transform leftArm;
    public Transform leftForeArm;
    public Transform leftHand;
    public Transform rightArm;
    public Transform rightForeArm;
    public Transform rightHand;

    // ---------------------------------------------------------------
    // Restricción de eje Z para brazos
    // ---------------------------------------------------------------
    [Header("Restricción de Ejes (Brazos)")]
    [Tooltip("Activa la restricción de eje Z en Unity (segunda capa de seguridad). " +
             "Python ya aplica la misma restricción; esta protege ante latencia de red.")]
    public bool enableArmZClamp = true;

    [Tooltip("Valor mínimo para Z de brazos en espacio Unity. " +
             "0 = no pueden ir hacia atrás. Valores negativos permiten algo de movimiento posterior.")]
    [Range(-0.5f, 0f)]
    public float armZClampMin = 0f;

    // ---------------------------------------------------------------
    // Suavizado
    // ---------------------------------------------------------------
    [Header("Suavizado de Movimiento")]
    [Tooltip("Velocidad de interpolación Slerp. Más alto = más rápido y menos suave. Rango recomendado: 8-20.")]
    [Range(1f, 30f)]
    public float smoothSpeed = 15f;

    // ---------------------------------------------------------------
    // Set de huesos que tienen restricción Z activa
    // ---------------------------------------------------------------
    private HashSet<Transform> armBones;

    private string lastProcessedData = "";
    private Dictionary<Transform, Quaternion> initialRotations = new Dictionary<Transform, Quaternion>();
    private Dictionary<Transform, Vector3>    initialDirections = new Dictionary<Transform, Vector3>();

    // ---------------------------------------------------------------
    // Unity lifecycle
    // ---------------------------------------------------------------

    void Start()
    {
        // Registrar qué transforms son "brazos" para aplicarles el clamp de Z
        armBones = new HashSet<Transform> { leftArm, leftForeArm, rightArm, rightForeArm };

        // Guardar la T-Pose inicial y la dirección hacia el hueso hijo
        SaveInitialState(leftArm,      leftForeArm);
        SaveInitialState(rightArm,     rightForeArm);
        SaveInitialState(leftForeArm,  leftHand);
        SaveInitialState(rightForeArm, rightHand);

        SaveInitialState(leftUpLeg,  leftLeg);
        SaveInitialState(rightUpLeg, rightLeg);
        SaveInitialState(leftLeg,    leftFoot);
        SaveInitialState(rightLeg,   rightFoot);
    }

    void Update()
    {
        if (udpClient == null) return;

        string currentData = udpClient.GetLastData();

        // Solo procesar si hay datos nuevos
        if (currentData != lastProcessedData && !string.IsNullOrEmpty(currentData))
        {
            lastProcessedData = currentData;
            ApplyPoses(currentData);
        }
    }

    // ---------------------------------------------------------------
    // Lógica de pose
    // ---------------------------------------------------------------

    void SaveInitialState(Transform bone, Transform child)
    {
        if (bone != null)
        {
            initialRotations[bone] = bone.rotation;
            if (child != null)
                initialDirections[bone] = (child.position - bone.position).normalized;
        }
    }

    void ApplyPoses(string jsonString)
    {
        try
        {
            JObject vectors = JObject.Parse(jsonString);

            // Brazos
            ApplyBoneTransform(leftArm,      vectors["mixamorig:LeftArm"]);
            ApplyBoneTransform(rightArm,     vectors["mixamorig:RightArm"]);
            ApplyBoneTransform(leftForeArm,  vectors["mixamorig:LeftForeArm"]);
            ApplyBoneTransform(rightForeArm, vectors["mixamorig:RightForeArm"]);

            // Piernas
            ApplyBoneTransform(leftUpLeg,  vectors["mixamorig:LeftUpLeg"]);
            ApplyBoneTransform(rightUpLeg, vectors["mixamorig:RightUpLeg"]);
            ApplyBoneTransform(leftLeg,    vectors["mixamorig:LeftLeg"]);
            ApplyBoneTransform(rightLeg,   vectors["mixamorig:RightLeg"]);
        }
        catch (System.Exception)
        {
            // Silenciar errores de parseo para no inundar la consola con paquetes UDP malformados.
            // Descomentar para depurar:
            // Debug.LogWarning("MixamoAnimator: paquete UDP malformado. " + e.Message);
        }
    }

    void ApplyBoneTransform(Transform bone, JToken data)
    {
        if (bone == null || data == null) return;
        if (!initialRotations.ContainsKey(bone) || !initialDirections.ContainsKey(bone)) return;

        float x = (float)data["x"];
        float y = (float)data["y"];
        float z = (float)data["z"];

        // --- Restricción de eje Z (segunda capa, Unity-side) ---
        // Python ya aplica el clamp; esto protege en caso de latencia
        // o si MixamoAnimator se usa sin el filtro de Python.
        if (enableArmZClamp && armBones != null && armBones.Contains(bone))
        {
            z = Mathf.Max(armZClampMin, z);

            // Re-normalizar después del clamp para mantener vector unitario
            float mag = Mathf.Sqrt(x * x + y * y + z * z);
            if (mag > 1e-6f) { x /= mag; y /= mag; z /= mag; }
        }

        Vector3 targetDir  = new Vector3(x, y, z);
        Vector3 initialDir = initialDirections[bone];

        // Si el vector tiene magnitud válida, aplicar rotación
        if (targetDir.sqrMagnitude > 1e-6f)
        {
            targetDir = targetDir.normalized;

            // Rotación desde la dirección T-Pose hacia la dirección recibida
            Quaternion rot           = Quaternion.FromToRotation(initialDir, targetDir);
            Quaternion targetRotation = rot * initialRotations[bone];

            // Slerp para suavizar (velocidad configurable desde el Inspector)
            bone.rotation = Quaternion.Slerp(bone.rotation, targetRotation, Time.deltaTime * smoothSpeed);
        }
        // Si el vector quedó en cero tras el clamp (caso extremo), mantenemos la rotación actual.
    }
}
