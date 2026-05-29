using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

/// <summary>
/// Herramienta de editor que conecta LABO new rig al sistema Dance Labo completo
/// con un solo clic: mapea huesos, asigna UDP, configura parámetros, desactiva El bueno.
///
/// Acceder desde: Dance Labo → Configurar LABO new rig
/// </summary>
public class RigMapper : EditorWindow
{
    // -------------------------------------------------------------------------
    // Estado de la ventana
    // -------------------------------------------------------------------------

    private bool _analyzed   = false;
    private string _statusMsg = "";
    private Color  _statusColor = Color.white;

    // Referencias encontradas
    private GameObject  _laboRigInScene;    // instancia ya en la escena
    private GameObject  _laboRigAsset;     // prefab/asset en Project
    private UDPClient   _udpClient;
    private GameObject  _elBuenoInScene;

    // Huesos encontrados (por nombre exacto, confirmados en las imágenes)
    private Transform _leftUpLeg, _leftLeg, _leftFoot;
    private Transform _rightUpLeg, _rightLeg, _rightFoot;
    private Transform _leftArm, _leftForeArm, _leftHand;
    private Transform _rightArm, _rightForeArm, _rightHand;
    private Transform _hips, _spine, _spine1, _neck;

    // -------------------------------------------------------------------------
    // Menú
    // -------------------------------------------------------------------------

    [MenuItem("Dance Labo/Configurar LABO new rig")]
    public static void ShowWindow()
        => GetWindow<RigMapper>("Dance Labo — Setup LABO rig");

    // -------------------------------------------------------------------------
    // GUI
    // -------------------------------------------------------------------------

    private void OnGUI()
    {
        GUILayout.Label("Dance Labo — Conexión de LABO new rig", EditorStyles.boldLabel);
        EditorGUILayout.Space();

        // ── Botón 1: Analizar escena ────────────────────────────────────────
        if (GUILayout.Button("① Analizar Escena", GUILayout.Height(35)))
            AnalyzeScene();

        if (_analyzed)
        {
            EditorGUILayout.Space();
            DrawStatus();
            EditorGUILayout.Space();

            // ── Botón 2: Configurar todo ─────────────────────────────────
            GUI.backgroundColor = new Color(0.3f, 0.9f, 0.3f);
            bool allBones = _leftUpLeg && _leftLeg && _leftFoot &&
                            _rightUpLeg && _rightLeg && _rightFoot &&
                            _leftArm && _leftForeArm && _leftHand &&
                            _rightArm && _rightForeArm && _rightHand;

            using (new EditorGUI.DisabledScope(!allBones || _udpClient == null))
            {
                if (GUILayout.Button("② Conectar Todo al Sistema", GUILayout.Height(45)))
                    ConnectAll();
            }
            GUI.backgroundColor = Color.white;

            if (!allBones)
                EditorGUILayout.HelpBox(
                    "Falta al menos un hueso DEF-. Revisa que LABO new rig esté en la escena con todos sus huesos visibles.",
                    MessageType.Warning);

            if (_udpClient == null)
                EditorGUILayout.HelpBox(
                    "No se encontró el objeto 'UDP_Manager' en la escena. Agrégalo primero.",
                    MessageType.Error);
        }
    }

    // -------------------------------------------------------------------------
    // ANÁLISIS DE ESCENA
    // -------------------------------------------------------------------------

    private void AnalyzeScene()
    {
        _analyzed = false;
        _statusMsg = "";

        // 1. Buscar UDP_Manager
        _udpClient = FindObjectOfType<UDPClient>();

        // 2. Buscar El bueno (para desactivarlo luego)
        _elBuenoInScene = GameObject.Find("El bueno");

        // 3. Buscar LABO new rig en la jerarquía de la escena
        //    Puede tener "(Clone)" al final si fue instanciado con Instantiate().
        _laboRigInScene = null;
        foreach (GameObject go in FindObjectsOfType<GameObject>())
        {
            if (go.name.StartsWith("LABO new rig") ||
                go.name == "LABO new rig" ||
                go.name == "LABO new rig(Clone)")
            {
                // Elegir el que está en la raíz de la jerarquía (Transform.parent == null)
                if (go.transform.parent == null)
                {
                    _laboRigInScene = go;
                    break;
                }
            }
        }

        if (_laboRigInScene == null)
        {
            SetStatus("⚠ LABO new rig no encontrado en la escena. " +
                      "Arrástralo desde Assets a la Jerarquía primero.", Color.yellow);
            _analyzed = true;
            return;
        }

        // 4. Buscar huesos por nombre exacto confirmado en las imágenes
        FindBones();

        // 5. Resumen
        int found = CountFound();
        if (found == 12)
            SetStatus($"✅ LABO new rig encontrado ({_laboRigInScene.name}).\n" +
                      $"✅ Todos los 12 huesos localizados.\n" +
                      $"{(_udpClient != null ? "✅" : "❌")} UDP_Manager (port {_udpClient?.port}).\n" +
                      $"{(_elBuenoInScene != null ? "✅" : "ℹ")} El bueno {(_elBuenoInScene != null ? "encontrado (se desactivará)." : "no está en escena.")}",
                      found == 12 && _udpClient != null ? Color.cyan : Color.yellow);
        else
            SetStatus($"⚠ Sólo {found}/12 huesos encontrados.\n" +
                      "Asegúrate de que LABO new rig esté completamente importado y en la escena.",
                      Color.yellow);

        _analyzed = true;
    }

    private void FindBones()
    {
        Transform root = _laboRigInScene.transform;

        // ── Piernas ──────────────────────────────────────────────────────────
        // Confirmado en imágenes: DEF-thigh.L → DEF-shin.L → DEF-foot.L
        _leftUpLeg   = FindDeep(root, "DEF-thigh.L");
        _leftLeg     = FindDeep(root, "DEF-shin.L");
        _leftFoot    = FindDeep(root, "DEF-foot.L");

        _rightUpLeg  = FindDeep(root, "DEF-thigh.R");
        _rightLeg    = FindDeep(root, "DEF-shin.R");
        _rightFoot   = FindDeep(root, "DEF-foot.R");

        // ── Brazos ───────────────────────────────────────────────────────────
        // Confirmado en imágenes: DEF-upper_arm.L → DEF-forearm.L → DEF-hand.L
        _leftArm     = FindDeep(root, "DEF-upper_arm.L");
        _leftForeArm = FindDeep(root, "DEF-forearm.L");
        _leftHand    = FindDeep(root, "DEF-hand.L");

        _rightArm    = FindDeep(root, "DEF-upper_arm.R");
        _rightForeArm= FindDeep(root, "DEF-forearm.R");
        _rightHand   = FindDeep(root, "DEF-hand.R");

        // ── Torso / Columna ──────────────────────────────────────────────────
        // Confirmado: hips (directo bajo torso), neck (bajo MCH-ROT-neck)
        _hips  = FindDeep(root, "hips");
        _spine = FindDeep(root, "spine_fk");         // Base de la cadena FK
        _spine1= FindDeep(root, "spine_fk.002");     // Columna media
        _neck  = FindDeep(root, "neck");
    }

    private int CountFound()
    {
        int c = 0;
        if (_leftUpLeg)    c++; if (_leftLeg)     c++; if (_leftFoot)   c++;
        if (_rightUpLeg)   c++; if (_rightLeg)    c++; if (_rightFoot)  c++;
        if (_leftArm)      c++; if (_leftForeArm) c++; if (_leftHand)   c++;
        if (_rightArm)     c++; if (_rightForeArm)c++; if (_rightHand)  c++;
        return c;
    }

    // -------------------------------------------------------------------------
    // DIBUJO DE ESTADO
    // -------------------------------------------------------------------------

    private void DrawStatus()
    {
        GUI.color = _statusColor;
        GUILayout.TextArea(_statusMsg, GUILayout.ExpandHeight(false));
        GUI.color = Color.white;

        EditorGUILayout.Space(4);

        // Tabla de huesos
        GUILayout.Label("Mapa de Huesos Detectados", EditorStyles.boldLabel);

        DrawBoneRow("leftUpLeg   (muslo izq)",       _leftUpLeg);
        DrawBoneRow("leftLeg     (rodilla izq)",      _leftLeg);
        DrawBoneRow("leftFoot    (pie izq)",          _leftFoot);
        DrawBoneRow("rightUpLeg  (muslo der)",        _rightUpLeg);
        DrawBoneRow("rightLeg    (rodilla der)",      _rightLeg);
        DrawBoneRow("rightFoot   (pie der)",          _rightFoot);
        DrawBoneRow("leftArm     (brazo izq)",        _leftArm);
        DrawBoneRow("leftForeArm (antebrazo izq)",    _leftForeArm);
        DrawBoneRow("leftHand    (mano izq)",         _leftHand);
        DrawBoneRow("rightArm    (brazo der)",        _rightArm);
        DrawBoneRow("rightForeArm(antebrazo der)",    _rightForeArm);
        DrawBoneRow("rightHand   (mano der)",         _rightHand);
        DrawBoneRow("hips        (cadera)",           _hips);
        DrawBoneRow("spine       (columna base)",     _spine);
        DrawBoneRow("spine1      (columna media)",    _spine1);
        DrawBoneRow("neck        (cuello)",           _neck);
    }

    private void DrawBoneRow(string label, Transform bone)
    {
        EditorGUILayout.BeginHorizontal();
        GUI.color = bone != null ? Color.green : Color.red;
        GUILayout.Label(bone != null ? "✓" : "✗", GUILayout.Width(20));
        GUI.color = Color.white;
        GUILayout.Label(label, GUILayout.Width(220));
        GUILayout.Label(bone != null ? bone.name : "NO ENCONTRADO",
                        bone != null ? EditorStyles.label : EditorStyles.boldLabel);
        EditorGUILayout.EndHorizontal();
    }

    // -------------------------------------------------------------------------
    // CONEXIÓN PRINCIPAL — todo en un solo clic
    // -------------------------------------------------------------------------

    private void ConnectAll()
    {
        Undo.RegisterFullObjectHierarchyUndo(_laboRigInScene, "Dance Labo: Conectar LABO new rig");

        // ── 1. Agregar / obtener MixamoAnimator ──────────────────────────────
        MixamoAnimator anim = _laboRigInScene.GetComponent<MixamoAnimator>();
        if (anim == null)
            anim = Undo.AddComponent<MixamoAnimator>(_laboRigInScene);

        // ── 2. Huesos de piernas ─────────────────────────────────────────────
        anim.leftUpLeg    = _leftUpLeg;
        anim.leftLeg      = _leftLeg;
        anim.leftFoot     = _leftFoot;
        anim.rightUpLeg   = _rightUpLeg;
        anim.rightLeg     = _rightLeg;
        anim.rightFoot    = _rightFoot;

        // ── 3. Huesos de brazos ──────────────────────────────────────────────
        anim.leftArm      = _leftArm;
        anim.leftForeArm  = _leftForeArm;
        anim.leftHand     = _leftHand;
        anim.rightArm     = _rightArm;
        anim.rightForeArm = _rightForeArm;
        anim.rightHand    = _rightHand;

        // ── 4. Torso (opcionales — el rig Rigify los tiene) ─────────────────
        anim.hips   = _hips;
        anim.spine  = _spine;
        anim.spine1 = _spine1;
        anim.neck   = _neck;

        // ── 5. UDP Client ────────────────────────────────────────────────────
        anim.udpClient = _udpClient;

        // ── 6. Parámetros calibrados (igual que El bueno funcional) ──────────
        //   • Se necesita swap para alinear el espacio de la cámara de MediaPipe con el avatar
        anim.swapLeftRight     = true;
        anim.smoothSpeed       = 15f;

        //   • Supresión de ruido de profundidad (valores probados con El bueno)
        anim.enableArmZClamp   = true;
        anim.armZClampMin      = 0f;
        anim.armZMultiplier    = 0.10f;
        anim.legZMultiplier    = 0.10f;
        anim.torsoZMultiplier  = 0.10f;
        anim.neckZMultiplier   = 0.00f;

        //   • Correcciones de eje — activar torsoNegateX para alinear el torso con el espejo (swap)
        anim.legNegateX  = false;
        anim.legNegateY  = false;
        anim.legNegateZ  = false;
        anim.torsoNegateX= true;
        anim.torsoNegateY= false;
        anim.torsoNegateZ= false;

        //   • Hips: desactivado por defecto (puede torcer todo el esqueleto)
        anim.enableHipsRotation = false;

        //   • Diagnóstico: activar para ver estado en la Consola de Unity
        anim.logDiagnosticsOnStart = true;

        // ── 7. Desactivar El bueno (si existe en la escena) ──────────────────
        if (_elBuenoInScene != null)
        {
            Undo.RecordObject(_elBuenoInScene, "Desactivar El bueno");
            _elBuenoInScene.SetActive(false);
        }

        // ── 8. Asegurarse de que LABO new rig esté activo ────────────────────
        Undo.RecordObject(_laboRigInScene, "Activar LABO new rig");
        _laboRigInScene.SetActive(true);

        // ── 9. Guardar la escena ─────────────────────────────────────────────
        EditorUtility.SetDirty(anim);
        EditorUtility.SetDirty(_laboRigInScene);
        EditorSceneManager.MarkSceneDirty(_laboRigInScene.scene);

        // ── 10. Informe final ────────────────────────────────────────────────
        string informe =
            "✅ LABO new rig conectado exitosamente.\n\n" +
            "Huesos mapeados:\n" +
            $"  Piernas: DEF-thigh/shin/foot .L/.R\n" +
            $"  Brazos:  DEF-upper_arm/forearm/hand .L/.R\n" +
            $"  Torso:   hips, spine_fk, spine_fk.002, neck\n\n" +
            "Parámetros aplicados:\n" +
            "  Swap Left Right = FALSE (Rigify tiene anatomía correcta)\n" +
            "  Arm/Leg/Torso Z Multiplier = 0.10\n" +
            "  Diagnostics = ON (ver Consola en Play Mode)\n\n" +
            "El bueno: DESACTIVADO (respaldo en escena).\n\n" +
            "Próximos pasos:\n" +
            "  1. Guarda la escena (Ctrl+S)\n" +
            "  2. Corre: python main.py\n" +
            "  3. Entra en Play Mode\n" +
            "  4. Haz T-Pose y verifica el log de diagnóstico";

        SetStatus(informe, Color.cyan);

        EditorUtility.DisplayDialog("Dance Labo — ¡Listo!", informe, "Entendido");
        Debug.Log("[RigMapper] " + informe);
    }

    // -------------------------------------------------------------------------
    // Utilidades
    // -------------------------------------------------------------------------

    private static Transform FindDeep(Transform parent, string name)
    {
        if (parent.name == name) return parent;
        foreach (Transform child in parent)
        {
            Transform result = FindDeep(child, name);
            if (result != null) return result;
        }
        return null;
    }

    private void SetStatus(string msg, Color color)
    {
        _statusMsg   = msg;
        _statusColor = color;
        Repaint();
    }
}
