/*
 * MovementComparator.cs — D-01 / D-02  (Dance Labo)
 *
 * Recibe los scores de similitud calculados por Python (LiveEvaluator + ScoringEngine)
 * a través del canal UDP y los expone para que FeedbackUI pueda mostrarlos.
 *
 * ARQUITECTURA:
 *   Python (LiveEvaluator) → JSON con __score_* → UDPClient.cs → MovementComparator.cs → FeedbackUI.cs
 *
 * CONFIGURACIÓN EN UNITY:
 *   1. Agregar este script al mismo GameObject que tiene UDPClient.cs
 *   2. En FeedbackUI, referenciar este MovementComparator para leer BodyPartScore
 *   3. Ajustar ThresholdGreen y ThresholdYellow desde el Inspector
 *
 * NOTA: MixamoAnimator ignora los campos __score_* del JSON; ambos scripts
 *       pueden coexistir en el mismo UDPClient sin conflictos.
 */

using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json.Linq;

[RequireComponent(typeof(UDPClient))]
public class MovementComparator : MonoBehaviour
{
    // ─── Inspector ────────────────────────────────────────────────────────────

    [Header("Umbrales de evaluación")]
    [Range(0f, 1f)]
    [Tooltip("Score >= este valor → Verde (bien replicado)")]
    public float ThresholdGreen  = 0.85f;

    [Range(0f, 1f)]
    [Tooltip("Score >= este valor → Amarillo (aceptable)")]
    public float ThresholdYellow = 0.60f;

    [Header("Diagnóstico")]
    [Tooltip("Score general 0–1 (calculado en Python)")]
    [Range(0f, 1f)] public float OverallScore = 0f;

    // ─── Datos públicos para FeedbackUI ──────────────────────────────────────

    /// <summary>
    /// Score por segmento corporal. Claves: brazo_izquierdo, brazo_derecho,
    /// pierna_izquierda, pierna_derecha, torso.
    /// Valores: 0.0 – 1.0  (calculados en Python por ScoringEngine).
    /// </summary>
    public Dictionary<string, float> BodyPartScore { get; private set; }
        = new Dictionary<string, float>
    {
        { "brazo_izquierdo",  0f },
        { "brazo_derecho",    0f },
        { "pierna_izquierda", 0f },
        { "pierna_derecha",   0f },
        { "torso",            0f },
    };

    // ─── Eventos (D-02) ──────────────────────────────────────────────────────

    /// <summary>Disparado cuando el score supera ThresholdGreen durante > 1 segundo.</summary>
    public event System.Action<float> OnReplicaDetected;

    /// <summary>Disparado cuando el estudiante lleva > 3 segundos consecutivos en verde.</summary>
    public event System.Action OnGoodStreak;

    /// <summary>Disparado cada frame con el nuevo score overall (0–1).</summary>
    public event System.Action<float> OnScoreUpdated;

    // ─── Privado ─────────────────────────────────────────────────────────────

    private UDPClient   _udp;
    private string      _lastData      = "";
    private float       _timeInGreen   = 0f;
    private bool        _streakFired   = false;
    private bool        _hasScore      = false;   // Python está enviando scores

    private static readonly string[] _segmentKeys =
    {
        "brazo_izquierdo", "brazo_derecho",
        "pierna_izquierda", "pierna_derecha",
        "torso"
    };
    private static readonly float[] _segmentWeights = { 0.25f, 0.25f, 0.20f, 0.20f, 0.10f };

    // ─── Unity lifecycle ─────────────────────────────────────────────────────

    void Start()
    {
        _udp = GetComponent<UDPClient>();
        if (_udp == null)
            Debug.LogError("[MovementComparator] Requiere un UDPClient en el mismo GameObject.");
    }

    void Update()
    {
        if (_udp == null) return;

        string data = _udp.GetLastData();
        if (string.IsNullOrEmpty(data) || data == _lastData) return;
        _lastData = data;

        ParseScorePacket(data);

        if (_hasScore)
            UpdateStreakLogic();
    }

    // ─── Parseo del JSON ─────────────────────────────────────────────────────

    private void ParseScorePacket(string json)
    {
        try
        {
            JObject obj = JObject.Parse(json);

            // El campo "__score_overall__" sólo viene cuando Python
            // tiene LiveEvaluator activo. Si no está, ignorar el paquete.
            JToken overallToken = obj["__score_overall__"];
            if (overallToken == null)
            {
                _hasScore = false;
                return;
            }

            _hasScore    = true;
            OverallScore = Mathf.Clamp01((float)overallToken);
            OnScoreUpdated?.Invoke(OverallScore);

            // Leer scores por segmento
            foreach (string seg in _segmentKeys)
            {
                JToken segToken = obj[$"__score_{seg}__"];
                if (segToken != null)
                    BodyPartScore[seg] = Mathf.Clamp01((float)segToken);
            }
        }
        catch
        {
            // JSON sin campos de score → paquete de pose normal, ignorar
            _hasScore = false;
        }
    }

    // ─── Lógica de racha (D-02) ──────────────────────────────────────────────

    private void UpdateStreakLogic()
    {
        if (OverallScore >= ThresholdGreen)
        {
            _timeInGreen += Time.deltaTime;

            // Evento: buena réplica detectada por > 1 segundo
            if (_timeInGreen >= 1f)
                OnReplicaDetected?.Invoke(OverallScore);

            // Evento: racha de 3 segundos consecutivos
            if (_timeInGreen >= 3f && !_streakFired)
            {
                OnGoodStreak?.Invoke();
                _streakFired = true;
                Debug.Log("[MovementComparator] ✅ GoodStreak — 3s en verde!");
            }
        }
        else
        {
            _timeInGreen = 0f;
            _streakFired = false;
        }
    }

    // ─── API pública ─────────────────────────────────────────────────────────

    /// <summary>
    /// Devuelve el score global ponderado (mismo cálculo que ScoringEngine en Python).
    /// </summary>
    public float GetOverallScore()
    {
        float total = 0f, weightTotal = 0f;
        for (int i = 0; i < _segmentKeys.Length; i++)
        {
            float w = _segmentWeights[i];
            total       += BodyPartScore[_segmentKeys[i]] * w;
            weightTotal += w;
        }
        return weightTotal > 0f ? total / weightTotal : 0f;
    }

    /// <summary>
    /// Devuelve "verde", "amarillo" o "rojo" para un segmento dado.
    /// Útil para FeedbackUI sin que tenga que conocer los umbrales.
    /// </summary>
    public string GetColorForSegment(string segment)
    {
        float s = BodyPartScore.ContainsKey(segment) ? BodyPartScore[segment] : 0f;
        if (s >= ThresholdGreen)  return "verde";
        if (s >= ThresholdYellow) return "amarillo";
        return "rojo";
    }

    /// <summary>True mientras Python está enviando scores activamente.</summary>
    public bool IsReceivingScores => _hasScore;
}
