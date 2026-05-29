using System.Collections.Generic;
using System.Text;
using UnityEngine;
using Newtonsoft.Json.Linq;

/// <summary>
/// Aplica poses recibidas por UDP (JSON con vectores de dirección) a un avatar humanoid.
/// Compatible con el modelo "El bueno" y con cualquier rig estilo Mixamo.
///
/// ARQUITECTURA:
///   Python/MediaPipe → JSON (vectores XYZ por hueso) → UDPClient → MixamoAnimator → Transforms del avatar
///
/// CLAVES JSON ESPERADAS DE PYTHON (sin cambios respecto a versión anterior):
///   Brazos : mixamorig:LeftArm, mixamorig:RightArm, mixamorig:LeftForeArm, mixamorig:RightForeArm
///   Piernas: mixamorig:LeftUpLeg, mixamorig:RightUpLeg, mixamorig:LeftLeg, mixamorig:RightLeg
///   Torso  : mixamorig:Hips, mixamorig:Spine, mixamorig:Spine1, mixamorig:Neck
///
/// MAPEO PARA "El bueno" (arrastrar Transform desde la jerarquía en el Inspector):
///   leftUpLeg   -> mixamorig:LeftLeg.L      leftLeg    -> mixamorig:LeftFoot.L
///   rightUpLeg  -> mixamorig:LeftLeg.R      rightLeg   -> mixamorig:LeftFoot.R
///   leftArm     -> mixamorig:LeftArm.L      leftForeArm -> mixamorig:LeftForeArm.L
///   rightArm    -> mixamorig:LeftArm.R      rightForeArm -> mixamorig:LeftForeArm.R
///   hips        -> mixamorig:Hips.Labo      spine       -> mixamorig:Spine1.Labo
///   spine1      -> mixamorig:Spine2.Labo    neck        -> mixamorig:Neck.Labo
/// </summary>
public class MixamoAnimator : MonoBehaviour
{
    // -------------------------------------------------------------------------
    // Inspector — Fuente de datos
    // -------------------------------------------------------------------------

    [Header("Fuente de Datos UDP")]
    public UDPClient udpClient;

    // -------------------------------------------------------------------------
    // Inspector — Huesos
    // -------------------------------------------------------------------------

    [Header("Piernas (UpperLeg = muslo, LowerLeg = tibia/rodilla)")]
    [Tooltip("'El bueno': mixamorig:LeftLeg.L  — Clave JSON: mixamorig:LeftUpLeg")]
    public Transform leftUpLeg;
    [Tooltip("'El bueno': mixamorig:LeftFoot.L — Clave JSON: mixamorig:LeftLeg")]
    public Transform leftLeg;
    [Tooltip("'El bueno': mixamorig:LeftToeBase.L (referencia, no animada directamente)")]
    public Transform leftFoot;
    [Tooltip("'El bueno': mixamorig:LeftLeg.R  — Clave JSON: mixamorig:RightUpLeg")]
    public Transform rightUpLeg;
    [Tooltip("'El bueno': mixamorig:LeftFoot.R — Clave JSON: mixamorig:RightLeg")]
    public Transform rightLeg;
    [Tooltip("'El bueno': mixamorig:LeftToeBase.R (referencia, no animada directamente)")]
    public Transform rightFoot;

    [Header("Brazos")]
    [Tooltip("'El bueno': mixamorig:LeftArm.L    — Clave JSON: mixamorig:LeftArm")]
    public Transform leftArm;
    [Tooltip("'El bueno': mixamorig:LeftForeArm.L — Clave JSON: mixamorig:LeftForeArm")]
    public Transform leftForeArm;
    [Tooltip("'El bueno': mixamorig:LeftHand.L")]
    public Transform leftHand;
    [Tooltip("'El bueno': mixamorig:LeftArm.R    — Clave JSON: mixamorig:RightArm")]
    public Transform rightArm;
    [Tooltip("'El bueno': mixamorig:LeftForeArm.R — Clave JSON: mixamorig:RightForeArm")]
    public Transform rightForeArm;
    [Tooltip("'El bueno': mixamorig:LeftHand.R")]
    public Transform rightHand;

    [Header("Torso / Columna (opcionales — si se dejan vacios el script los ignora)")]
    [Tooltip("'El bueno': mixamorig:Hips.Labo    — Clave JSON: mixamorig:Hips")]
    public Transform hips;
    [Tooltip("'El bueno': mixamorig:Spine1.Labo  — Clave JSON: mixamorig:Spine")]
    public Transform spine;
    [Tooltip("'El bueno': mixamorig:Spine2.Labo  — Clave JSON: mixamorig:Spine1")]
    public Transform spine1;
    [Tooltip("'El bueno': mixamorig:Neck.Labo    — Clave JSON: mixamorig:Neck")]
    public Transform neck;

    // -------------------------------------------------------------------------
    // Inspector — Restriccion de eje Z (brazos)
    // -------------------------------------------------------------------------

    [Header("Restriccion de Ejes (Brazos)")]
    [Tooltip("Activa restriccion de eje Z en Unity. Python ya la aplica; esta es segunda capa de seguridad.")]
    public bool enableArmZClamp = true;

    [Tooltip("Minimo para Z de brazos. 0 = no van hacia atras. Negativo permite algo de movimiento posterior.")]
    [Range(-0.5f, 0f)]
    public float armZClampMin = 0f;

    [Tooltip("Factor de escala para el eje Z de los brazos (0 = sin profundidad, 1 = completo).\n" +
             "MediaPipe sobre-estima la profundidad al subir brazos (z~0.95 en vez de ~0).\n" +
             "Usa 0.0-0.15 para que Y domine y los brazos suban correctamente.")]
    [Range(0f, 1f)]
    public float armZMultiplier = 0.10f;

    // -------------------------------------------------------------------------
    // Inspector — Correccion de ejes
    // -------------------------------------------------------------------------

    [Header("Correccion de Ejes — Piernas")]
    [Tooltip("Negar X del vector para TODAS las piernas. Activar si doblan en direccion contraria en X.")]
    public bool legNegateX = false;
    [Tooltip("Negar Y del vector para TODAS las piernas.")]
    public bool legNegateY = false;
    [Tooltip("Negar Z del vector para TODAS las piernas.")]
    public bool legNegateZ = false;

    [Tooltip("Factor de escala para el eje Z de las piernas (0 = sin profundidad, 1 = completo).\n" +
             "MediaPipe estima mal la profundidad: usa 0.0-0.2 para reducir ruido.\n" +
             "Ajustar en tiempo real hasta que las piernas no se doblen hacia atras.")]
    [Range(0f, 1f)]
    public float legZMultiplier = 0.10f;

    [Header("Mapeo de Extremidades (Espejo)")]
    [Tooltip("Intercambia los lados izquierdo y derecho al recibir datos UDP.\n" +
             "Activar para rigs como 'El bueno' donde los huesos .L (pantalla-izquierda) y .R (pantalla-derecha) estan cruzados anatomicamente.")]
    public bool swapLeftRight = false;

    [Header("Correccion de Ejes — Torso / Cadera")]
    [Tooltip("Negar X del vector para huesos de torso (Hips, Spine, Neck).")]
    public bool torsoNegateX = false;
    [Tooltip("Negar Y del vector para huesos de torso.")]
    public bool torsoNegateY = false;
    [Tooltip("Negar Z del vector para huesos de torso.")]
    public bool torsoNegateZ = false;

    [Tooltip("Factor de escala para el eje Z del torso (Hips, Spine, Spine1) (0 = sin profundidad, 1 = completo).\n" +
             "Reducir para atenuar ruido de profundidad y evitar que se incline demasiado.")]
    [Range(0f, 1f)]
    public float torsoZMultiplier = 0.10f;

    [Tooltip("Factor de escala para el eje Z del cuello (Neck) (0 = sin profundidad, 1 = completo).\n" +
             "Se recomienda usar 0 o 0.05 para evitar que la cabeza se doble hacia adelante/atras por ruido.")]
    [Range(0f, 1f)]
    public float neckZMultiplier = 0.00f;

    // -------------------------------------------------------------------------
    // Inspector — Suavizado
    // -------------------------------------------------------------------------

    [Header("Suavizado de Movimiento")]
    [Tooltip("Velocidad de interpolacion Slerp. Rango recomendado: 8-20. Mas alto = mas rapido.")]
    [Range(1f, 30f)]
    public float smoothSpeed = 15f;

    // -------------------------------------------------------------------------
    // Inspector — Diagnostico
    // -------------------------------------------------------------------------

    [Header("Diagnostico")]
    [Tooltip("Imprime estado de todos los huesos asignados en la consola al entrar en Play Mode.")]
    public bool logDiagnosticsOnStart = false;

    [Tooltip("Activa la rotacion del hueso Hips (raiz del esqueleto).\n" +
             "DESACTIVAR si el personaje aparece deformado al agregar torso.\n" +
             "El hueso Hips es el padre de todo — si sus datos de Python son incorrectos\n" +
             "dobla todo el esqueleto. Spine/Spine1/Neck se animan igual sin necesidad de Hips.")]
    public bool enableHipsRotation = false;

    // -------------------------------------------------------------------------
    // Privado
    // -------------------------------------------------------------------------

    private enum BoneGroup { Arm, Leg, Torso }

    private HashSet<Transform> _armBones;

    private string _lastProcessedData = "";
    private readonly Dictionary<Transform, Quaternion> _initialRotations  = new Dictionary<Transform, Quaternion>();
    private readonly Dictionary<Transform, Vector3>    _initialDirections = new Dictionary<Transform, Vector3>();

    // -------------------------------------------------------------------------
    // Unity lifecycle
    // -------------------------------------------------------------------------

    void Start()
    {
        _armBones = new HashSet<Transform> { leftArm, leftForeArm, rightArm, rightForeArm };

        // Brazos
        SaveInitialState(leftArm,      leftForeArm);
        SaveInitialState(rightArm,     rightForeArm);
        SaveInitialState(leftForeArm,  leftHand);
        SaveInitialState(rightForeArm, rightHand);

        // Piernas
        SaveInitialState(leftUpLeg,  leftLeg);
        SaveInitialState(rightUpLeg, rightLeg);
        SaveInitialState(leftLeg,    leftFoot);
        SaveInitialState(rightLeg,   rightFoot);

        // Torso — cadena: hips -> spine -> spine1 -> neck
        // Si spine1 o neck son null, se usa el siguiente disponible como child de referencia
        SaveInitialState(hips,   spine  != null ? spine  : neck);
        SaveInitialState(spine,  spine1 != null ? spine1 : neck);
        SaveInitialState(spine1, neck);
        SaveInitialState(neck,   null);  // neck no requiere child

        if (logDiagnosticsOnStart)
            LogDiagnostics();
    }

    void LateUpdate()
    {
        if (udpClient == null) return;

        string currentData = udpClient.GetLastData();

        if (currentData != _lastProcessedData && !string.IsNullOrEmpty(currentData))
        {
            _lastProcessedData = currentData;
            ApplyPoses(currentData);

            if (logDiagnosticsOnStart)
            {
                try
                {
                    JObject v = JObject.Parse(currentData);
                    Debug.Log(
                        $"[Torso] Hips:{v["mixamorig:Hips"]?.ToString(Newtonsoft.Json.Formatting.None)}" +
                        $" | Spine:{v["mixamorig:Spine"]?.ToString(Newtonsoft.Json.Formatting.None)}" +
                        $" | Neck:{v["mixamorig:Neck"]?.ToString(Newtonsoft.Json.Formatting.None)}"
                    );
                    Debug.Log(
                        $"[Legs] LeftUpLeg:{v["mixamorig:LeftUpLeg"]?.ToString(Newtonsoft.Json.Formatting.None)}" +
                        $" | LeftLeg:{v["mixamorig:LeftLeg"]?.ToString(Newtonsoft.Json.Formatting.None)}"
                    );
                    Debug.Log(
                        $"[Arms] LeftArm:{v["mixamorig:LeftArm"]?.ToString(Newtonsoft.Json.Formatting.None)}" +
                        $" | RightArm:{v["mixamorig:RightArm"]?.ToString(Newtonsoft.Json.Formatting.None)}" +
                        $" | LeftForeArm:{v["mixamorig:LeftForeArm"]?.ToString(Newtonsoft.Json.Formatting.None)}" +
                        $" | RightForeArm:{v["mixamorig:RightForeArm"]?.ToString(Newtonsoft.Json.Formatting.None)}"
                    );
                }
                catch {}
            }
        }
    }

    // -------------------------------------------------------------------------
    // Logica principal de pose
    // -------------------------------------------------------------------------

    /// <summary>
    /// Guarda rotacion T-Pose y direccion hacia el hijo para cada hueso.
    /// Si child es null usa bone.up como fallback (evita NullRef en huesos terminales).
    /// Si el hueso ya fue registrado, no sobreescribe (evita doble llamada).
    /// </summary>
    private void SaveInitialState(Transform bone, Transform child)
    {
        if (bone == null) return;
        if (_initialRotations.ContainsKey(bone)) return;

        _initialRotations[bone] = bone.rotation;
        _initialDirections[bone] = child != null
            ? (child.position - bone.position).normalized
            : bone.up;
    }

    /// <summary>
    /// Parsea el JSON UDP y aplica rotaciones a todos los huesos registrados.
    /// Las claves JSON son identicas a las que Python ya envia (sin cambios en Python).
    /// Los Transforms del Inspector apuntan a los huesos del nuevo rig "El bueno".
    /// </summary>
    private void ApplyPoses(string jsonString)
    {
        try
        {
            JObject v = JObject.Parse(jsonString);

            // ── ORDEN IMPORTANTE: raiz → hojas ────────────────────────────────────
            // Los huesos de torso (.Labo) son padres de los huesos de extremidades.
            // Si se aplicaran DESPUES que piernas/brazos, la rotacion del padre
            // invalida la world rotation ya seteada en los hijos.
            // Siempre: torso primero → despues piernas y brazos.
            // ──────────────────────────────────────────────────────────────────────

            // 1. Torso / Columna (padres — se aplican primero)
            // Hips es la raiz del esqueleto — solo se anima si enableHipsRotation = true.
            // Si los datos de Python para Hips son incorrectos, dobla TODO el personaje.
            if (enableHipsRotation)
                ApplyBoneTransform(hips, v["mixamorig:Hips"], BoneGroup.Torso);
            ApplyBoneTransform(spine,  v["mixamorig:Spine"],  BoneGroup.Torso);
            ApplyBoneTransform(spine1, v["mixamorig:Spine1"], BoneGroup.Torso);
            ApplyBoneTransform(neck,   v["mixamorig:Neck"],   BoneGroup.Torso);

            // 2. Piernas (hijos del torso)
            JToken lUpLeg = swapLeftRight ? v["mixamorig:RightUpLeg"] : v["mixamorig:LeftUpLeg"];
            JToken rUpLeg = swapLeftRight ? v["mixamorig:LeftUpLeg"]  : v["mixamorig:RightUpLeg"];
            JToken lLeg   = swapLeftRight ? v["mixamorig:RightLeg"]   : v["mixamorig:LeftLeg"];
            JToken rLeg   = swapLeftRight ? v["mixamorig:LeftLeg"]    : v["mixamorig:RightLeg"];

            ApplyBoneTransform(leftUpLeg,  lUpLeg,  BoneGroup.Leg);
            ApplyBoneTransform(rightUpLeg, rUpLeg, BoneGroup.Leg);
            ApplyBoneTransform(leftLeg,    lLeg,    BoneGroup.Leg);
            ApplyBoneTransform(rightLeg,   rLeg,   BoneGroup.Leg);

            // 3. Brazos (hijos del torso)
            JToken lArm    = swapLeftRight ? v["mixamorig:RightArm"]     : v["mixamorig:LeftArm"];
            JToken rArm    = swapLeftRight ? v["mixamorig:LeftArm"]      : v["mixamorig:RightArm"];
            JToken lForeArm = swapLeftRight ? v["mixamorig:RightForeArm"] : v["mixamorig:LeftForeArm"];
            JToken rForeArm = swapLeftRight ? v["mixamorig:LeftForeArm"]  : v["mixamorig:RightForeArm"];

            ApplyBoneTransform(leftArm,      lArm,      BoneGroup.Arm);
            ApplyBoneTransform(rightArm,     rArm,      BoneGroup.Arm);
            ApplyBoneTransform(leftForeArm,  lForeArm,  BoneGroup.Arm);
            ApplyBoneTransform(rightForeArm, rForeArm, BoneGroup.Arm);
        }
        catch (System.Exception)
        {
            // Silenciar errores de parseo — paquetes UDP malformados o __score_* del MovementComparator.
            // Descomentar para depurar:
            // Debug.LogWarning("[MixamoAnimator] paquete malformado: " + e.Message);
        }
    }

    /// <summary>
    /// Aplica rotacion calculada por FromToRotation + Slerp a un hueso.
    /// Aplica correcciones de eje segun el grupo (Leg / Torso / Arm).
    /// </summary>
    private void ApplyBoneTransform(Transform bone, JToken data, BoneGroup group)
    {
        if (bone == null || data == null) return;
        if (!_initialRotations.ContainsKey(bone) || !_initialDirections.ContainsKey(bone)) return;

        float x = (float)data["x"];
        float y = (float)data["y"];
        float z = (float)data["z"];

        // Correcciones de eje por grupo
        switch (group)
        {
            case BoneGroup.Leg:
                if (legNegateX) x = -x;
                if (legNegateY) y = -y;
                if (legNegateZ) z = -z;
                // Amortiguacion de profundidad: MediaPipe estima Z con mucho ruido.
                // Un valor alto de Z hace que las piernas se doblen hacia atras.
                // Reducir con este multiplicador (0.0-0.2 recomendado).
                z *= legZMultiplier;
                break;

            case BoneGroup.Torso:
                if (torsoNegateX) x = -x;
                if (torsoNegateY) y = -y;
                if (torsoNegateZ) z = -z;
                if (bone == neck)
                    z *= neckZMultiplier;
                else
                    z *= torsoZMultiplier;
                break;

            case BoneGroup.Arm:
                // 1. Amortiguar profundidad ruidosa de MediaPipe (igual que piernas)
                //    Al subir los brazos, MediaPipe genera z~0.95 en vez de ~0,
                //    haciendo que el avatar rote los brazos hacia la camara en vez de arriba.
                z *= armZMultiplier;

                // 2. Clamp de Z (segunda capa): impedir que vayan detras del cuerpo
                if (enableArmZClamp && _armBones != null && _armBones.Contains(bone))
                {
                    z = Mathf.Max(armZClampMin, z);
                    float mag = Mathf.Sqrt(x * x + y * y + z * z);
                    if (mag > 1e-6f) { x /= mag; y /= mag; z /= mag; }
                }
                break;
        }

        Vector3 targetDir  = new Vector3(x, y, z);
        Vector3 initialDir = _initialDirections[bone];

        if (targetDir.sqrMagnitude > 1e-6f)
        {
            targetDir = targetDir.normalized;

            // Rotacion desde la direccion T-Pose hacia la direccion recibida
            Quaternion rot            = Quaternion.FromToRotation(initialDir, targetDir);
            Quaternion targetRotation = rot * _initialRotations[bone];

            // Slerp para suavizar
            bone.rotation = Quaternion.Slerp(bone.rotation, targetRotation, Time.deltaTime * smoothSpeed);
        }
        // Si el vector quedo en cero tras el clamp, mantener rotacion actual.
    }

    // -------------------------------------------------------------------------
    // Diagnostico
    // -------------------------------------------------------------------------

    private void LogDiagnostics()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("<color=cyan>[MixamoAnimator] === Diagnostico de Huesos ===</color>");

        LogBone(sb, "leftUpLeg  (mixamorig:LeftUpLeg)",      leftUpLeg);
        LogBone(sb, "leftLeg    (mixamorig:LeftLeg)",        leftLeg);
        LogBone(sb, "leftFoot   (referencia)",               leftFoot);
        LogBone(sb, "rightUpLeg (mixamorig:RightUpLeg)",     rightUpLeg);
        LogBone(sb, "rightLeg   (mixamorig:RightLeg)",       rightLeg);
        LogBone(sb, "rightFoot  (referencia)",               rightFoot);
        LogBone(sb, "leftArm    (mixamorig:LeftArm)",        leftArm);
        LogBone(sb, "leftForeArm(mixamorig:LeftForeArm)",    leftForeArm);
        LogBone(sb, "leftHand   (referencia)",               leftHand);
        LogBone(sb, "rightArm   (mixamorig:RightArm)",       rightArm);
        LogBone(sb, "rightForeArm(mixamorig:RightForeArm)",  rightForeArm);
        LogBone(sb, "rightHand  (referencia)",               rightHand);
        LogBone(sb, "hips       (mixamorig:Hips)",           hips);
        LogBone(sb, "spine      (mixamorig:Spine)",          spine);
        LogBone(sb, "spine1     (mixamorig:Spine1)",         spine1);
        LogBone(sb, "neck       (mixamorig:Neck)",           neck);

        if (udpClient == null)
            sb.AppendLine("<color=red>  WARNING: udpClient NO asignado — el avatar no recibira datos UDP.</color>");
        else
            sb.AppendLine($"<color=green>  OK: udpClient -> {udpClient.host}:{udpClient.port}</color>");

        Debug.Log(sb.ToString());
    }

    private void LogBone(StringBuilder sb, string label, Transform bone)
    {
        if (bone == null)
        {
            sb.AppendLine($"  <color=yellow>? {label}: [NO ASIGNADO]</color>");
            return;
        }
        bool hasDir = _initialDirections.ContainsKey(bone);
        string dir  = hasDir ? _initialDirections[bone].ToString("F3") : "sin hijo valido";
        sb.AppendLine($"  OK {label}: <b>{bone.name}</b> | T-Pose dir: {dir}");
    }
}
