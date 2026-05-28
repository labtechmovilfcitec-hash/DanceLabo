/*
 * FeedbackUI.cs — D-05 / D-02  (Dance Labo)
 *
 * Muestra en tiempo real qué partes del cuerpo están bien posicionadas,
 * conectándose directamente a MovementComparator para leer los scores
 * enviados desde Python (LiveEvaluator + ScoringEngine).
 *
 * CONFIGURACIÓN EN UNITY:
 *   1. Crear un Canvas en la escena (Screen Space - Overlay).
 *   2. Agregar este script a un GameObject vacío dentro del Canvas.
 *   3. En el Inspector, asignar las referencias a los paneles UI.
 *   4. Asignar MovementComparator al campo "Comparator".
 *
 * LAYOUT RECOMENDADO:
 *   Canvas
 *   └── FeedbackUI (este script)
 *       ├── ScorePanel       → Panel con TxtScoreGlobal y TxtLabel
 *       ├── SegmentPanel     → Panel lateral con los 5 indicadores
 *       │   ├── ImgBrazoIzq  (Image, color cambia con lerp)
 *       │   ├── ImgBrazoDer
 *       │   ├── ImgPiernaIzq
 *       │   ├── ImgPiernaDer
 *       │   └── ImgTorso
 *       └── FinalPanel       → Se activa al terminar secuencia
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class FeedbackUI : MonoBehaviour
{
    // ─── Inspector — Referencias ──────────────────────────────────────────────

    [Header("Fuente de datos")]
    [Tooltip("MovementComparator que recibe los scores desde Python")]
    public MovementComparator Comparator;

    [Header("Panel de score global")]
    public TextMeshProUGUI TxtScoreGlobal;  // "87%"
    public TextMeshProUGUI TxtLabel;        // "Excelente / Bien / Sigue practicando"
    public Image           ImgScoreBar;     // Barra de progreso del score

    [Header("Indicadores por segmento")]
    public Image ImgBrazoIzq;
    public Image ImgBrazoDer;
    public Image ImgPiernaIzq;
    public Image ImgPiernaDer;
    public Image ImgTorso;

    [Header("Labels de segmento (opcional)")]
    public TextMeshProUGUI TxtBrazoIzq;
    public TextMeshProUGUI TxtBrazoDer;
    public TextMeshProUGUI TxtPiernaIzq;
    public TextMeshProUGUI TxtPiernaDer;
    public TextMeshProUGUI TxtTorso;

    [Header("Panel de resultado final")]
    [Tooltip("Se activa al llamar ShowFinalResult()")]
    public GameObject FinalPanel;
    public TextMeshProUGUI TxtFinalScore;
    public TextMeshProUGUI TxtFinalLabel;

    [Header("Animación")]
    [Range(1f, 20f)]
    [Tooltip("Velocidad del lerp de colores (mayor = más rápido)")]
    public float LerpSpeed = 6f;

    // ─── Colores ──────────────────────────────────────────────────────────────

    private static readonly Color ColorVerde    = new Color(0.18f, 0.80f, 0.44f); // #2ecc71
    private static readonly Color ColorAmarillo = new Color(0.95f, 0.61f, 0.07f); // #f39c12
    private static readonly Color ColorRojo     = new Color(0.91f, 0.30f, 0.24f); // #e74c3c
    private static readonly Color ColorGris     = new Color(0.30f, 0.30f, 0.35f); // inactivo

    // ─── Estado interno ───────────────────────────────────────────────────────

    private float _displayedScore = 0f;        // Score que se muestra (suavizado)
    private readonly Dictionary<string, Image> _segmentImages
        = new Dictionary<string, Image>();
    private readonly Dictionary<string, TextMeshProUGUI> _segmentTexts
        = new Dictionary<string, TextMeshProUGUI>();

    // ─── Unity lifecycle ─────────────────────────────────────────────────────

    void Start()
    {
        // Mapear segmentos a sus imágenes y textos
        _segmentImages["brazo_izquierdo"]  = ImgBrazoIzq;
        _segmentImages["brazo_derecho"]    = ImgBrazoDer;
        _segmentImages["pierna_izquierda"] = ImgPiernaIzq;
        _segmentImages["pierna_derecha"]   = ImgPiernaDer;
        _segmentImages["torso"]            = ImgTorso;

        _segmentTexts["brazo_izquierdo"]   = TxtBrazoIzq;
        _segmentTexts["brazo_derecho"]     = TxtBrazoDer;
        _segmentTexts["pierna_izquierda"]  = TxtPiernaIzq;
        _segmentTexts["pierna_derecha"]    = TxtPiernaDer;
        _segmentTexts["torso"]             = TxtTorso;

        // Estado inicial: todo gris (sin datos)
        SetAllSegmentsColor(ColorGris);
        if (FinalPanel) FinalPanel.SetActive(false);

        // Suscribirse a eventos del comparador
        if (Comparator != null)
        {
            Comparator.OnScoreUpdated    += HandleScoreUpdated;
            Comparator.OnReplicaDetected += HandleReplicaDetected;
            Comparator.OnGoodStreak      += HandleGoodStreak;
        }
        else
        {
            Debug.LogWarning("[FeedbackUI] No hay MovementComparator asignado. " +
                             "Los scores no se mostrarán.");
        }
    }

    void Update()
    {
        if (Comparator == null || !Comparator.IsReceivingScores)
        {
            SetAllSegmentsColor(Color.Lerp(_segmentImages["torso"]?.color ?? ColorGris,
                                           ColorGris, Time.deltaTime * LerpSpeed));
            return;
        }

        // Animar score global con lerp
        float targetScore = Comparator.GetOverallScore();
        _displayedScore = Mathf.Lerp(_displayedScore, targetScore, Time.deltaTime * LerpSpeed);

        UpdateScoreDisplay(_displayedScore);
        UpdateSegmentDisplay();
    }

    void OnDestroy()
    {
        if (Comparator != null)
        {
            Comparator.OnScoreUpdated    -= HandleScoreUpdated;
            Comparator.OnReplicaDetected -= HandleReplicaDetected;
            Comparator.OnGoodStreak      -= HandleGoodStreak;
        }
    }

    // ─── Actualización de UI ──────────────────────────────────────────────────

    private void UpdateScoreDisplay(float score)
    {
        int pct = Mathf.RoundToInt(score * 100f);
        Color targetColor = ScoreToColor(score);

        if (TxtScoreGlobal)
            TxtScoreGlobal.text = $"{pct}%";

        if (TxtLabel)
            TxtLabel.text = ScoreToLabel(score);

        if (TxtScoreGlobal)
            TxtScoreGlobal.color = Color.Lerp(TxtScoreGlobal.color, targetColor, Time.deltaTime * LerpSpeed);

        // Barra de progreso
        if (ImgScoreBar)
        {
            ImgScoreBar.fillAmount = Mathf.Lerp(ImgScoreBar.fillAmount, score, Time.deltaTime * LerpSpeed);
            ImgScoreBar.color      = Color.Lerp(ImgScoreBar.color, targetColor, Time.deltaTime * LerpSpeed);
        }
    }

    private void UpdateSegmentDisplay()
    {
        foreach (var kvp in _segmentImages)
        {
            string seg = kvp.Key;
            Image img  = kvp.Value;
            if (img == null) continue;

            float segScore = Comparator.BodyPartScore.ContainsKey(seg)
                ? Comparator.BodyPartScore[seg] : 0f;
            Color target = ScoreToColor(segScore);

            // Lerp suave de color (sin parpadeo)
            img.color = Color.Lerp(img.color, target, Time.deltaTime * LerpSpeed);

            // Actualizar texto del segmento si existe
            if (_segmentTexts.TryGetValue(seg, out TextMeshProUGUI txt) && txt != null)
            {
                txt.text  = $"{Mathf.RoundToInt(segScore * 100f)}%";
                txt.color = Color.Lerp(txt.color, target, Time.deltaTime * LerpSpeed);
            }
        }
    }

    // ─── Panel de resultado final ─────────────────────────────────────────────

    /// <summary>
    /// Muestra el panel de resultado final con score y desglose.
    /// Llamar desde SequencePlayer al terminar la secuencia.
    /// </summary>
    public void ShowFinalResult()
    {
        if (FinalPanel == null) return;
        FinalPanel.SetActive(true);

        float score = Comparator != null ? Comparator.GetOverallScore() : _displayedScore;
        if (TxtFinalScore) TxtFinalScore.text  = $"{Mathf.RoundToInt(score * 100f)}%";
        if (TxtFinalLabel) TxtFinalLabel.text  = ScoreToLabel(score);
        if (TxtFinalScore) TxtFinalScore.color = ScoreToColor(score);

        StartCoroutine(HideFinalPanelAfterDelay(5f));
    }

    private IEnumerator HideFinalPanelAfterDelay(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        if (FinalPanel) FinalPanel.SetActive(false);
    }

    // ─── Eventos del comparador ───────────────────────────────────────────────

    private void HandleScoreUpdated(float score)
    {
        // El update de UI se hace en Update() para el lerp. Este evento es para
        // lógica adicional que quiera reaccionar al nuevo score.
    }

    private void HandleReplicaDetected(float score)
    {
        // Puedes agregar efectos, sonidos, etc. aquí
        Debug.Log($"[FeedbackUI] ✅ Réplica detectada — score: {Mathf.RoundToInt(score * 100f)}%");
    }

    private void HandleGoodStreak()
    {
        Debug.Log("[FeedbackUI] 🔥 ¡Racha de 3 segundos en verde!");
        // Aquí podrías disparar un efecto de partículas, sonido, etc.
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private Color ScoreToColor(float score)
    {
        if (Comparator != null)
        {
            if (score >= Comparator.ThresholdGreen)  return ColorVerde;
            if (score >= Comparator.ThresholdYellow) return ColorAmarillo;
            return ColorRojo;
        }
        if (score >= 0.85f) return ColorVerde;
        if (score >= 0.60f) return ColorAmarillo;
        return ColorRojo;
    }

    private static string ScoreToLabel(float score)
    {
        if (score >= 0.85f) return "¡Excelente!";
        if (score >= 0.60f) return "¡Bien!";
        return "Sigue practicando";
    }

    private void SetAllSegmentsColor(Color color)
    {
        foreach (var img in _segmentImages.Values)
            if (img) img.color = color;
    }
}
