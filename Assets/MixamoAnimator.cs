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
    [Tooltip("Hueso de la cabeza (opcional — si no se asigna, busca 'head' automáticamente en los hijos de neck)")]
    public Transform head;

    [Header("Rotación Corporal (Body Yaw)")]
    [Tooltip("Activa la rotación horizontal del cuerpo completo cuando la persona gira en la vida real.\n" +
             "Python envía 'body_yaw' en grados y el avatar rota en Y replicando el giro 360°.")]
    public bool enableBodyYaw = true;

    [Tooltip("Tiempo de suavizado orgánico del giro corporal (segundos). Rango: 0.15 - 0.25 para rotación humana fluida.")]
    [Range(0.05f, 0.6f)]
    public float yawSmoothTime = 0.20f;

    [Tooltip("Invierte la rotación en Y si el avatar gira en dirección opuesta a ti.")]
    public bool invertYaw = false;

    [Tooltip("Ajuste fino de ángulo en grados (normalmente 0).")]
    [Range(-180f, 180f)]
    public float yawOffset = 0f;

    // -------------------------------------------------------------------------
    // Inspector — Restriccion de eje Z (brazos)
    // -------------------------------------------------------------------------

    [Header("Restriccion de Ejes (Brazos)")]
    [Tooltip("Activa restriccion de eje Z en Unity en el espacio local del cuerpo para evitar que los brazos se claven detrás de la espalda.")]
    public bool enableArmZClamp = true;

    [Tooltip("Mínimo para Z local de brazos. -0.15 permite rango natural sin clavarse detrás del cuerpo.")]
    [Range(-0.5f, 0.1f)]
    public float armZClampMin = -0.15f;

    [Tooltip("Factor de escala para el alcance frontal/profundidad de los brazos en el espacio del personaje.\n" +
             "0.6 - 1.0 permite estirar los brazos al frente de forma natural.")]
    [Range(0f, 1.5f)]
    public float armZMultiplier = 0.85f;

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

    // -------------------------------------------------------------------------
    // Inspector — Suavizado y Rendimiento
    // -------------------------------------------------------------------------

    [Header("Suavizado y Rendimiento")]
    [Tooltip("Velocidad de interpolación y seguimiento. Rango recomendado: 20-35. Mayor valor = movimiento más rápido y directo.")]
    [Range(1f, 60f)]
    public float smoothSpeed = 30f;

    [Tooltip("Aceleración dinámica: cuando el modelo hace movimientos rápidos o cambios bruscos, aumenta la velocidad para que no vaya rezagado ni lento.")]
    public bool responsiveSpeedBoost = true;

    [Tooltip("Activa la interpolación continua y fluida en cada frame (60+ FPS). Si se desactiva, el modelo saltará directamente a la última pose recibida.")]
    public bool enableSmoothing = true;

    [Tooltip("Fuerza la tasa de refresco mínima a 60 FPS en Unity para evitar tirones o bajones de fluidez visual.")]
    public bool optimizeFrameRate = true;

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
    private int    _lastProcessedVersion = -1;
    private bool   _hasReceivedFirstPacket = false;
    private Transform  _actualRoot;
    private Quaternion _initialRootRotation = Quaternion.identity;
    private float      _currentYaw = 0f;
    private float      _targetYaw  = 0f;
    private float      _yawVelocity = 0f;
    private readonly Dictionary<Transform, Quaternion> _initialRotations  = new Dictionary<Transform, Quaternion>();
    private readonly Dictionary<Transform, Vector3>    _initialDirections = new Dictionary<Transform, Vector3>();
    private readonly Dictionary<Transform, Quaternion> _targetRotations   = new Dictionary<Transform, Quaternion>();
    private readonly List<Transform>                  _orderedBones      = new List<Transform>();
    private Transform _spineFk;
    private Transform _spine1Fk;

    // -------------------------------------------------------------------------
    // Unity lifecycle
    // -------------------------------------------------------------------------

    void Start()
    {
        _actualRoot = transform;
        _initialRootRotation = _actualRoot.rotation;

        if (udpClient == null)
        {
            udpClient = FindObjectOfType<UDPClient>();
            if (udpClient != null)
                Debug.Log("<color=green>[MixamoAnimator] UDPClient encontrado y vinculado automáticamente.</color>");
            else
                Debug.LogWarning("[MixamoAnimator] No se encontró ningún componente UDPClient en la escena.");
        }

        if (optimizeFrameRate && Application.targetFrameRate < 60)
        {
            Application.targetFrameRate = 60;
        }

        _armBones = new HashSet<Transform> { leftArm, leftForeArm, leftHand, rightArm, rightForeArm, rightHand };

        // Configuración anatómica para LABO new rig (alienígena sin cuello largo separado):
        // En este alienígena, la mandíbula inferior y lengua están en DEF-spine.005 y el cráneo en DEF-spine.006.
        // Si neck y head se rotan por separado, la boca se desgarra/abre grotescamente.
        // Al asignar head a DEF-spine.005 (la base de la cabeza) y desactivar neck independiente,
        // toda la cabeza (cráneo + ojos + mandíbula) rota en un solo bloque sólido sin deformar la boca.
        Transform defHeadBase = FindBoneDeep(transform, "DEF-spine.005");
        if (defHeadBase != null)
        {
            head = defHeadBase;
            neck = null;
            Debug.Log($"<color=cyan>[MixamoAnimator] Hueso head fijado a '{head.name}' (base sólida cráneo+mandíbula)</color>");

            Transform defSpine = FindBoneDeep(transform, "DEF-spine.002") ?? FindBoneDeep(transform, "DEF-spine.001");
            if (defSpine != null)
            {
                spine = defSpine;
                Debug.Log($"<color=cyan>[MixamoAnimator] Hueso spine fijado a '{spine.name}'</color>");
            }

            Transform defSpine1 = FindBoneDeep(transform, "DEF-spine.003") ?? FindBoneDeep(transform, "DEF-spine.004");
            if (defSpine1 != null)
            {
                spine1 = defSpine1;
                Debug.Log($"<color=cyan>[MixamoAnimator] Hueso spine1 fijado a '{spine1.name}'</color>");
            }
        }
        else if (head == null || head.name.StartsWith("MCH-") || head.name.StartsWith("WGT-"))
        {
            head = FindBoneDeep(transform, "DEF-spine.006") 
                ?? FindBoneDeep(transform, "head") 
                ?? FindBoneDeep(transform, "DEF-head");
        }

        // Si neck es un hueso MCH- o WGT- y no es LABO, corregir al deform/FK real
        if (defHeadBase == null && (neck == null || neck.name.StartsWith("MCH-") || neck.name.StartsWith("WGT-")))
        {
            Transform foundNeck = FindBoneDeep(transform, "DEF-spine.005") 
                               ?? FindBoneDeep(transform, "neck") 
                               ?? FindBoneDeep(transform, "DEF-neck");
            if (foundNeck != null)
            {
                neck = foundNeck;
                Debug.Log($"<color=cyan>[MixamoAnimator] Hueso neck asignado a hueso real: '{neck.name}'</color>");
            }
        }

        // Torso Deform Chain: DEF-spine.002 (abdomen) y DEF-spine.003 (pecho)
        // Permite reverencia (saludo japonés) y agacharse moviendo todo el cuerpo sin separar la cabeza
        if (spine == null || spine.name.StartsWith("MCH-") || spine.name.StartsWith("WGT-") || spine.name == "spine_fk")
        {
            Transform defSpine = FindBoneDeep(transform, "DEF-spine.002") ?? FindBoneDeep(transform, "DEF-spine.001");
            if (defSpine != null)
            {
                spine = defSpine;
                Debug.Log($"<color=cyan>[MixamoAnimator] Hueso spine fijado a '{spine.name}'</color>");
            }
        }

        if (spine1 == null || spine1.name.StartsWith("MCH-") || spine1.name.StartsWith("WGT-") || spine1.name == "spine_fk.002")
        {
            Transform defSpine1 = FindBoneDeep(transform, "DEF-spine.003") ?? FindBoneDeep(transform, "DEF-spine.004");
            if (defSpine1 != null)
            {
                spine1 = defSpine1;
                Debug.Log($"<color=cyan>[MixamoAnimator] Hueso spine1 fijado a '{spine1.name}'</color>");
            }
        }

        if (hips == null || hips.name.StartsWith("MCH-") || hips.name.StartsWith("WGT-"))
        {
            Transform defHips = FindBoneDeep(transform, "DEF-spine");
            if (defHips != null) hips = defHips;
        }

        // Sincronización de hombros/brazos en Rigify (LABO):
        // En Rigify, los hombros y brazos cuelgan de MCH-spine.002 / spine_fk.002.
        // Al sincronizar _spineFk con la rotación de spine, tanto la malla del pecho como los hombros y brazos
        // se inclinan juntos hacia adelante durante una reverencia sin separarse.
        _spineFk = FindBoneDeep(transform, "MCH-spine.002") ?? FindBoneDeep(transform, "spine_fk.002") ?? FindBoneDeep(transform, "torso");
        _spine1Fk = FindBoneDeep(transform, "MCH-spine.003") ?? FindBoneDeep(transform, "spine_fk.003");
        if (_spineFk != null)
        {
            SaveInitialState(_spineFk, _spine1Fk);
        }
        if (_spine1Fk != null)
        {
            SaveInitialState(_spine1Fk, null);
        }

        // Calibración para Rigify (LABO new rig) en modo espejo (selfie):
        // 1. swapLeftRight = true alinea los huesos .L (X<0) con RightArm (X<0) y .R (X>0) con LeftArm (X>0)
        // 2. armZMultiplier = 1.0f permite extender los brazos hacia el frente con alcance 100% natural
        // 3. torsoZMultiplier = 1.0f permite inclinarse hacia adelante en reverencia/saludo japonés
        // 4. armZClampMin = -0.25f impide que los brazos se claven detrás del cuerpo
        swapLeftRight        = true;
        torsoNegateX         = false;
        enableArmZClamp      = true;
        armZClampMin         = -0.25f;
        armZMultiplier       = 1.0f;
        torsoZMultiplier     = 1.0f;
        enableBodyYaw        = true;
        enableHipsRotation   = false;

        // Brazos y Manos
        SaveInitialState(leftArm,      leftForeArm);
        SaveInitialState(rightArm,     rightForeArm);
        SaveInitialState(leftForeArm,  leftHand);
        SaveInitialState(rightForeArm, rightHand);
        SaveInitialState(leftHand,     null);
        SaveInitialState(rightHand,    null);

        // Piernas
        SaveInitialState(leftUpLeg,  leftLeg);
        SaveInitialState(rightUpLeg, rightLeg);
        SaveInitialState(leftLeg,    leftFoot);
        SaveInitialState(rightLeg,   rightFoot);

        // Torso — cadena: hips -> spine -> spine1 -> neck -> head
        SaveInitialState(hips,   spine  != null ? spine  : (neck != null ? neck : head));
        SaveInitialState(spine,  spine1 != null ? spine1 : (neck != null ? neck : head));
        SaveInitialState(spine1, neck   != null ? neck   : head);
        if (neck != null)
            SaveInitialState(neck, head);
        if (head != null)
            SaveInitialState(head, head.childCount > 0 ? head.GetChild(0) : null);

        BuildOrderedBoneList();

        if (logDiagnosticsOnStart)
            LogDiagnostics();
    }

    void LateUpdate()
    {
        if (udpClient != null)
        {
            int currentVer = udpClient.DataVersion;
            if (currentVer != _lastProcessedVersion)
            {
                _lastProcessedVersion = currentVer;
                string currentData = udpClient.GetLastData();

                if (!string.IsNullOrEmpty(currentData) && currentData.Contains("mixamorig:"))
                {
                    if (!_hasReceivedFirstPacket)
                    {
                        _hasReceivedFirstPacket = true;
                        Debug.Log("<color=lime>[MixamoAnimator] ¡Primer paquete de pose recibido y procesado correctamente!</color>");
                    }

                    ApplyPoses(currentData);

                    if (logDiagnosticsOnStart)
                    {
                        LogCurrentPose(currentData);
                    }
                }
            }
        }

        // Suavizado continuo de rotación del cuerpo completo (Body Yaw) en cada frame con inercia orgánica (SmoothDamp)
        if (enableBodyYaw && _actualRoot != null)
        {
            float angleDelta = Mathf.Abs(Mathf.DeltaAngle(_currentYaw, _targetYaw));
            // Aceleración dinámica para giros y vueltas rápidas:
            // Si la persona gira rápidamente (> 25° de diferencia), reduce el smoothTime para responder al instante sin rezago
            float effectiveYawSmooth = (responsiveSpeedBoost && angleDelta > 25f)
                ? Mathf.Max(0.06f, yawSmoothTime * 0.35f)
                : yawSmoothTime;

            _currentYaw = Mathf.SmoothDampAngle(_currentYaw, _targetYaw, ref _yawVelocity, effectiveYawSmooth);
            _actualRoot.rotation = _initialRootRotation * Quaternion.Euler(0f, _currentYaw, 0f);
        }

        // Interpolación continua CADA FRAME (a 60, 120 o 144 FPS)
        // Esto elimina el retraso y el efecto 'robotico/lento' entre paquetes UDP
        InterpolateBones();
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
        Vector3 dir = child != null ? (child.position - bone.position).normalized : Vector3.zero;
        if (dir.sqrMagnitude < 1e-4f)
        {
            if (bone.childCount > 0)
                dir = (bone.GetChild(0).position - bone.position).normalized;
            if (dir.sqrMagnitude < 1e-4f && bone.parent != null)
                dir = (bone.position - bone.parent.position).normalized;
            if (dir.sqrMagnitude < 1e-4f)
                dir = bone.up;
        }
        _initialDirections[bone] = dir;
    }

    /// <summary>
    /// Parsea el JSON UDP y aplica rotaciones a todos los huesos registrados.
    /// </summary>
    private void ApplyPoses(string jsonString)
    {
        try
        {
            JObject v = JObject.Parse(jsonString);

            // 0. Body Yaw (Giro corporal 360°)
            if (enableBodyYaw && v["body_yaw"] != null)
            {
                float rawYaw = (float)v["body_yaw"];
                if (invertYaw) rawYaw = -rawYaw;
                _targetYaw = rawYaw + yawOffset;
            }

            // 1. Torso / Columna (padres — se aplican primero)
            if (enableHipsRotation)
                ApplyBoneTransform(hips, v["mixamorig:Hips"], BoneGroup.Torso);
            ApplyBoneTransform(spine,  v["mixamorig:Spine"],  BoneGroup.Torso);
            if (_spineFk != null && _spineFk != spine)
            {
                ApplyBoneTransform(_spineFk, v["mixamorig:Spine"], BoneGroup.Torso);
            }
            ApplyBoneTransform(spine1, v["mixamorig:Spine1"], BoneGroup.Torso);
            if (_spine1Fk != null && _spine1Fk != spine1)
            {
                ApplyBoneTransform(_spine1Fk, v["mixamorig:Spine1"], BoneGroup.Torso);
            }
            // 0b. Detección de orientación espaldas (Direct Mode vs Mirror Mode)
            bool isBackward = v["is_backward"] != null && (bool)v["is_backward"];
            bool shouldSwap = swapLeftRight && !isBackward;

            // 1b. Rotación 3D de Cabeza / Cuello (Yaw, Pitch, Roll)
            JToken hr = v["head_rotation"];
            if (hr != null)
            {
                float hPitch = (float)hr["pitch"];
                float hYaw   = (float)hr["yaw"];
                float hRoll  = (float)hr["roll"];

                // Si shouldSwap está activo (espejo), invertimos Yaw y Roll para inclinar y rotar al mismo lado
                if (shouldSwap)
                {
                    hYaw  = -hYaw;
                    hRoll = -hRoll;
                }

                Transform targetHead = head != null ? head : neck;
                if (targetHead != null && _initialRotations.ContainsKey(targetHead))
                {
                    // La cabeza acompaña e integra la inclinación lateral del torso (DEF-spine.003 / DEF-spine.002)
                    Transform torsoRef = spine1 != null ? spine1 : spine;
                    Quaternion torsoTilt = Quaternion.identity;
                    float torsoRollDeg = 0f;
                    if (torsoRef != null && _targetRotations.TryGetValue(torsoRef, out Quaternion spineRot) && _initialRotations.ContainsKey(torsoRef))
                    {
                        torsoTilt = spineRot * Quaternion.Inverse(_initialRotations[torsoRef]);
                        if (enableBodyYaw)
                        {
                            torsoTilt = Quaternion.Euler(0f, -_currentYaw, 0f) * torsoTilt;
                        }
                        // Extraer inclinación lateral (roll en Z) del torso para no duplicarla con el roll de los ojos
                        torsoRollDeg = torsoTilt.eulerAngles.z;
                        if (torsoRollDeg > 180f) torsoRollDeg -= 360f;
                    }

                    // Inclinación relativa de la cabeza sobre el cuello (evita duplicar el roll de hombros):
                    // Si el usuario inclina todo el cuerpo, el torso rota y la cabeza acompaña naturalmente
                    float relativeRoll = Mathf.Clamp(hRoll - torsoRollDeg, -25f, 25f);

                    Quaternion headOffset = Quaternion.Euler(hPitch, hYaw, -relativeRoll);
                    Quaternion targetRot = torsoTilt * headOffset * _initialRotations[targetHead];
                    if (enableBodyYaw)
                    {
                        targetRot = Quaternion.Euler(0f, _currentYaw, 0f) * targetRot;
                    }
                    _targetRotations[targetHead] = targetRot;

                    // Cuello: aplica 35% de la rotación para un movimiento suave y orgánico de todo el cuello
                    if (neck != null && neck != targetHead && _initialRotations.ContainsKey(neck))
                    {
                        Quaternion neckOffset = Quaternion.Euler(hPitch * 0.35f, hYaw * 0.35f, -relativeRoll * 0.35f);
                        Quaternion neckTargetRot = torsoTilt * neckOffset * _initialRotations[neck];
                        if (enableBodyYaw)
                        {
                            neckTargetRot = Quaternion.Euler(0f, _currentYaw, 0f) * neckTargetRot;
                        }
                        _targetRotations[neck] = neckTargetRot;
                    }
                }
            }
            else
            {
                ApplyBoneTransform(neck, v["mixamorig:Neck"], BoneGroup.Torso);
            }

            // 2. Piernas (hijos del torso)
            JToken lUpLeg = shouldSwap ? v["mixamorig:RightUpLeg"] : v["mixamorig:LeftUpLeg"];
            JToken rUpLeg = shouldSwap ? v["mixamorig:LeftUpLeg"]  : v["mixamorig:RightUpLeg"];
            JToken lLeg   = shouldSwap ? v["mixamorig:RightLeg"]   : v["mixamorig:LeftLeg"];
            JToken rLeg   = shouldSwap ? v["mixamorig:LeftLeg"]    : v["mixamorig:RightLeg"];

            ApplyBoneTransform(leftUpLeg,  lUpLeg,  BoneGroup.Leg);
            ApplyBoneTransform(rightUpLeg, rUpLeg, BoneGroup.Leg);
            ApplyBoneTransform(leftLeg,    lLeg,    BoneGroup.Leg);
            ApplyBoneTransform(rightLeg,   rLeg,   BoneGroup.Leg);

            // 3. Brazos y Manos (hijos del torso)
            JToken lArm     = shouldSwap ? v["mixamorig:RightArm"]     : v["mixamorig:LeftArm"];
            JToken rArm     = shouldSwap ? v["mixamorig:LeftArm"]      : v["mixamorig:RightArm"];
            JToken lForeArm = shouldSwap ? v["mixamorig:RightForeArm"] : v["mixamorig:LeftForeArm"];
            JToken rForeArm = shouldSwap ? v["mixamorig:LeftForeArm"]  : v["mixamorig:RightForeArm"];
            JToken lHand    = shouldSwap ? v["mixamorig:RightHand"]    : v["mixamorig:LeftHand"];
            JToken rHand    = shouldSwap ? v["mixamorig:LeftHand"]     : v["mixamorig:RightHand"];

            ApplyBoneTransform(leftArm,      lArm,      BoneGroup.Arm);
            ApplyBoneTransform(rightArm,     rArm,      BoneGroup.Arm);
            ApplyBoneTransform(leftForeArm,  lForeArm,  BoneGroup.Arm);
            ApplyBoneTransform(rightForeArm, rForeArm, BoneGroup.Arm);
            ApplyBoneTransform(leftHand,     lHand,     BoneGroup.Arm);
            ApplyBoneTransform(rightHand,    rHand,     BoneGroup.Arm);
        }
        catch (System.Exception)
        {
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
        }

        Vector3 targetDir  = new Vector3(x, y, z);
        Vector3 initialDir = _initialDirections[bone];

        if (targetDir.sqrMagnitude > 1e-6f)
        {
            targetDir = targetDir.normalized;

            // 1. Transformar el vector recibido (espacio de cámara) al espacio LOCAL del personaje
            Vector3 localDir = targetDir;
            if (enableBodyYaw)
            {
                localDir = Quaternion.Euler(0f, -_currentYaw, 0f) * targetDir;
            }

            // 2. En el espacio local del personaje (+Z es frente al pecho, -Z es espalda):
            if (group == BoneGroup.Arm)
            {
                localDir.z *= armZMultiplier;

                // Restricción Z local: impide que los brazos se claven detrás de la espalda
                // sin importar en qué dirección esté girado el personaje en el mundo
                if (enableArmZClamp && _armBones != null && _armBones.Contains(bone))
                {
                    localDir.z = Mathf.Max(armZClampMin, localDir.z);
                }

                if (localDir.sqrMagnitude > 1e-6f)
                    localDir = localDir.normalized;
            }

            // 2b. Restricción anatómica para huesos de columna (Spine, Spine1, _spineFk):
            // Flexión hacia adelante (reverencia/saludo japonés): permitida de forma completa hasta 60°.
            // Inclinación lateral (Roll izquierda/derecha): permitida de forma fluida y continua hasta 55°.
            // Hiperextensión hacia atrás: limitada para impedir colapsos o quiebres hacia atrás.
            if (group == BoneGroup.Torso && (bone == spine || bone == spine1 || bone == _spineFk || bone == _spine1Fk))
            {
                // Limitar hyperextensión hacia atrás en Z (-Z en espacio local es hacia atrás)
                if (localDir.z < -0.12f)
                {
                    localDir.z = -0.12f;
                    if (localDir.sqrMagnitude > 1e-6f)
                        localDir = localDir.normalized;
                }

                float angle = Vector3.Angle(initialDir, localDir);
                // Si la persona se inclina hacia atrás (Z < initialDir.z - 0.10f) sin inclinación lateral (|X| < 0.20f),
                // limitar a 18° para no quebrar la columna hacia atrás.
                // En flexión frontal (reverencia) o inclinación lateral, se permite el rango completo de 60° sin bloqueos bruscos.
                bool isPureBackwardLean = localDir.z < (initialDir.z - 0.10f) && Mathf.Abs(localDir.x) < 0.20f;
                float maxAllowedAngle = isPureBackwardLean ? 18.0f : 60.0f;

                if (angle > maxAllowedAngle)
                {
                    localDir = Vector3.RotateTowards(initialDir, localDir, maxAllowedAngle * Mathf.Deg2Rad, 0f);
                }
                if (localDir.sqrMagnitude > 1e-6f)
                    localDir = localDir.normalized;
            }

            // 2c. Restricción anatómica para piernas (UpLeg y Leg):
            // Limitar la desviación angular máxima (75° para muslos UpLeg, 60° para piernas Leg)
            // para que las patitas alienígenas se flexionen y eleven con naturalidad al bailar o agacharse
            // sin retorcerse ni atravesar la malla.
            if (group == BoneGroup.Leg)
            {
                float maxLegAngle = (bone == leftUpLeg || bone == rightUpLeg) ? 75.0f : 60.0f;
                float legAngle = Vector3.Angle(initialDir, localDir);
                if (legAngle > maxLegAngle)
                {
                    localDir = Vector3.RotateTowards(initialDir, localDir, maxLegAngle * Mathf.Deg2Rad, 0f);
                }
                if (localDir.sqrMagnitude > 1e-6f)
                    localDir = localDir.normalized;
            }

            // 3. Rotación local desde la pose T-Pose inicial hacia la dirección local deseada
            Quaternion rot = Quaternion.FromToRotation(initialDir, localDir);
            Quaternion localTargetRotation = rot * _initialRotations[bone];

            // 4. Re-aplicar el yaw del cuerpo para obtener la rotación final de mundo del hueso
            Quaternion worldTargetRotation = localTargetRotation;
            if (enableBodyYaw)
            {
                worldTargetRotation = Quaternion.Euler(0f, _currentYaw, 0f) * localTargetRotation;
            }

            // Para manos: limitar la deflexión respecto a su postura natural de reposo alineada con el antebrazo.
            // Esto permite todos los movimientos de la Macarena y baile sin que la muñeca salte o se quiebre.
            if (bone == leftHand || bone == rightHand)
            {
                Transform parentArm = bone == leftHand ? leftForeArm : rightForeArm;
                if (parentArm != null && _targetRotations.TryGetValue(parentArm, out Quaternion parentTarget)
                    && _initialRotations.ContainsKey(parentArm) && _initialRotations.ContainsKey(bone))
                {
                    // Postura natural de reposo de la mano en el espacio de mundo actual del antebrazo
                    Quaternion initialRelRot = Quaternion.Inverse(_initialRotations[parentArm]) * _initialRotations[bone];
                    Quaternion naturalHandRot = parentTarget * initialRelRot;

                    float wristDeflection = Quaternion.Angle(naturalHandRot, worldTargetRotation);
                    if (wristDeflection > 65.0f)
                    {
                        worldTargetRotation = Quaternion.RotateTowards(naturalHandRot, worldTargetRotation, 65.0f);
                    }
                }
            }

            // Almacenar rotación objetivo para interpolación continua en cada frame
            _targetRotations[bone] = worldTargetRotation;
        }
        // Si el vector quedo en cero tras el clamp, mantener rotacion actual.
    }

    /// <summary>
    /// Interpola suavemente las rotaciones de todos los huesos activos en orden jerárquico (Torso -> Piernas -> Brazos).
    /// Se ejecuta en cada frame independientemente del intervalo de paquetes UDP para lograr 60+ FPS de fluidez.
    /// </summary>
    private void InterpolateBones()
    {
        float dt = Time.deltaTime;
        if (dt <= 0f) return;

        int count = _orderedBones.Count;
        for (int i = 0; i < count; i++)
        {
            Transform bone = _orderedBones[i];
            if (bone == null) continue;
            if (bone == hips && !enableHipsRotation) continue;

            if (_targetRotations.TryGetValue(bone, out Quaternion targetRot))
            {
                if (!enableSmoothing || smoothSpeed >= 55f)
                {
                    bone.rotation = targetRot;
                    continue;
                }

                float effectiveSpeed = smoothSpeed;

                // Aceleración dinámica para movimientos rápidos:
                // Si la distancia angular supera los 15 grados, aumentamos dinámicamente la velocidad
                // para que el avatar no se quede atrás en bailes o movimientos veloces.
                if (responsiveSpeedBoost)
                {
                    float angle = Quaternion.Angle(bone.rotation, targetRot);
                    if (angle > 15f)
                    {
                        float factor = Mathf.Clamp01((angle - 15f) / 60f);
                        effectiveSpeed = Mathf.Lerp(smoothSpeed, smoothSpeed * 2.5f, factor);
                    }
                }

                // Suavizado exponencial independiente de la tasa de cuadros (dt)
                float t = 1f - Mathf.Exp(-effectiveSpeed * dt);
                bone.rotation = Quaternion.Slerp(bone.rotation, targetRot, t);
            }
        }
    }

    private void BuildOrderedBoneList()
    {
        _orderedBones.Clear();

        // 1. Torso primero (padres)
        AddBoneToOrder(hips);
        AddBoneToOrder(spine);
        if (_spineFk != null) AddBoneToOrder(_spineFk);
        AddBoneToOrder(spine1);
        if (_spine1Fk != null) AddBoneToOrder(_spine1Fk);
        AddBoneToOrder(neck);
        if (head != null) AddBoneToOrder(head);

        // 2. Piernas (hijos)
        AddBoneToOrder(leftUpLeg);
        AddBoneToOrder(rightUpLeg);
        AddBoneToOrder(leftLeg);
        AddBoneToOrder(rightLeg);

        // 3. Brazos y Manos (hijos)
        AddBoneToOrder(leftArm);
        AddBoneToOrder(rightArm);
        AddBoneToOrder(leftForeArm);
        AddBoneToOrder(rightForeArm);
        AddBoneToOrder(leftHand);
        AddBoneToOrder(rightHand);
    }

    private void AddBoneToOrder(Transform bone)
    {
        if (bone != null && !_orderedBones.Contains(bone))
        {
            _orderedBones.Add(bone);
            if (!_targetRotations.ContainsKey(bone))
            {
                _targetRotations[bone] = bone.rotation;
            }
        }
    }

    private void LogCurrentPose(string currentData)
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
        LogBone(sb, "head       (mixamorig:Head)",           head);

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

    private Transform FindBoneDeep(Transform parent, string boneName)
    {
        if (parent == null) return null;
        if (parent.name.Equals(boneName, System.StringComparison.OrdinalIgnoreCase)) return parent;
        int count = parent.childCount;
        for (int i = 0; i < count; i++)
        {
            Transform found = FindBoneDeep(parent.GetChild(i), boneName);
            if (found != null) return found;
        }
        return null;
    }
}
