"""
scoring_engine.py — D-06: Algoritmo de Puntuación Acumulada
Dance Labo · Sprint 2

Compara la pose del estudiante (MediaPipe live) vs la pose de referencia
generada por el MotionLSTMGenerator, frame a frame, por segmento corporal.
"""

import numpy as np
import json
import os
from datetime import datetime


# ── Definición de segmentos corporales (índices de MediaPipe landmarks) ──────
BODY_SEGMENTS = {
    "torso": [11, 12, 23, 24],          # hombros + caderas
    "brazo_izquierdo": [13, 15, 17, 19, 21],   # codo, muñeca, dedos izq
    "brazo_derecho": [14, 16, 18, 20, 22],      # codo, muñeca, dedos der
    "pierna_izquierda": [25, 27, 29, 31],       # rodilla, tobillo, talón, pie izq
    "pierna_derecha": [26, 28, 30, 32],         # rodilla, tobillo, talón, pie der
    "cabeza": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # nariz, ojos, orejas, boca
}

# Peso de cada segmento en el score general (suma = 1.0)
SEGMENT_WEIGHTS = {
    "torso": 0.25,
    "brazo_izquierdo": 0.20,
    "brazo_derecho": 0.20,
    "pierna_izquierda": 0.15,
    "pierna_derecha": 0.15,
    "cabeza": 0.05,
}

# Umbrales de evaluación
THRESHOLD_VERDE   = 0.85  # >= 85% → Verde (bien)
THRESHOLD_AMARILLO = 0.60  # >= 60% → Amarillo (regular)
# < 60% → Rojo (mal)

HISTORIAL_FILE = "data/historial_scores.json"


def landmarks_dict_to_vector(landmarks_dict: dict) -> np.ndarray:
    """
    Convierte el dict de landmarks de MediaPipe ({id: {x,y,z}})
    al vector plano de 99 valores que usa el modelo LSTM.
    """
    vector = np.zeros(99, dtype=np.float32)
    for i in range(33):
        lm = landmarks_dict.get(str(i)) or landmarks_dict.get(i)
        if lm:
            vector[i * 3]     = lm.get('x', 0.0)
            vector[i * 3 + 1] = lm.get('y', 0.0)
            vector[i * 3 + 2] = lm.get('z', 0.0)
    return vector


def lstm_tensor_to_vector(frame_tensor) -> np.ndarray:
    """
    Convierte el frame de salida del LSTM (tensor de 99 dims)
    a un np.array. Acepta tensor PyTorch o np.array.
    """
    if hasattr(frame_tensor, 'detach'):
        return frame_tensor.detach().cpu().numpy().flatten()
    return np.array(frame_tensor, dtype=np.float32).flatten()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similitud coseno entre dos vectores. Retorna valor en [0, 1]."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos = np.dot(v1, v2) / (norm1 * norm2)
    # Mapear de [-1,1] a [0,1]
    return float((cos + 1.0) / 2.0)


def segment_similarity(
    student_vector: np.ndarray,
    reference_vector: np.ndarray,
    segment_indices: list
) -> float:
    """
    Calcula la similitud coseno entre estudiante y referencia
    solo para los landmarks del segmento dado.
    """
    seg_student = np.concatenate([student_vector[i*3:(i+1)*3] for i in segment_indices])
    seg_reference = np.concatenate([reference_vector[i*3:(i+1)*3] for i in segment_indices])
    return cosine_similarity(seg_student, seg_reference)


def score_label(score: float) -> str:
    """Retorna el texto de evaluación según el score."""
    if score >= THRESHOLD_VERDE:
        return "Excelente"
    elif score >= THRESHOLD_AMARILLO:
        return "Bien"
    else:
        return "Sigue practicando"


def color_for_score(score: float) -> str:
    """Retorna el color para la FeedbackUI."""
    if score >= THRESHOLD_VERDE:
        return "verde"
    elif score >= THRESHOLD_AMARILLO:
        return "amarillo"
    else:
        return "rojo"


# ── Clase principal ────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Motor de puntuación acumulada para el sistema de enseñanza Dance Labo.

    Uso típico (por frame):
        engine = ScoringEngine()
        engine.start_session("secuencia_01")
        for frame in frames:
            result = engine.evaluate_frame(student_landmarks, reference_vector)
            # result["overall"]  → float 0-1
            # result["segments"] → {"torso": 0.9, "brazo_izquierdo": 0.7, ...}
            # result["colors"]   → {"torso": "verde", ...}
        final = engine.end_session()
        # final["score_pct"]   → 74.3
        # final["desglose"]    → dict por segmento
        # final["label"]       → "Bien"
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia la sesión actual."""
        self._frame_scores = []           # lista de dicts de score por frame
        self._segment_accumulators = {seg: [] for seg in BODY_SEGMENTS}
        self._session_active = False
        self._movement_name = ""
        self._start_time = None

    def start_session(self, movement_name: str):
        """Inicia una nueva sesión de evaluación."""
        self.reset()
        self._session_active = True
        self._movement_name = movement_name
        self._start_time = datetime.now()
        print(f"[ScoringEngine] Sesión iniciada: {movement_name}")

    def evaluate_frame(
        self,
        student_landmarks: dict,
        reference_frame_tensor
    ) -> dict:
        """
        Evalúa un frame comparando la pose del estudiante vs la referencia.

        Args:
            student_landmarks: dict de MediaPipe {id: {x,y,z,v}}
            reference_frame_tensor: tensor/array (99,) del LSTM generativo

        Returns:
            dict con:
              - "overall": float [0,1] — score global del frame
              - "segments": dict por segmento, float [0,1]
              - "colors": dict por segmento, str ("verde"/"amarillo"/"rojo")
              - "overall_color": str
        """
        if not self._session_active:
            return {}

        student_vec = landmarks_dict_to_vector(student_landmarks)
        reference_vec = lstm_tensor_to_vector(reference_frame_tensor)

        # Calcular similitud por segmento
        segment_scores = {}
        for seg_name, indices in BODY_SEGMENTS.items():
            sim = segment_similarity(student_vec, reference_vec, indices)
            segment_scores[seg_name] = sim
            self._segment_accumulators[seg_name].append(sim)

        # Score global ponderado
        overall = sum(
            segment_scores[seg] * SEGMENT_WEIGHTS[seg]
            for seg in BODY_SEGMENTS
        )

        # Colores para FeedbackUI
        colors = {seg: color_for_score(s) for seg, s in segment_scores.items()}

        result = {
            "overall": overall,
            "segments": segment_scores,
            "colors": colors,
            "overall_color": color_for_score(overall),
        }
        self._frame_scores.append(result)
        return result

    def end_session(self) -> dict:
        """
        Finaliza la sesión y calcula el resultado final acumulado.

        Returns:
            dict con:
              - "score_pct": float — porcentaje 0-100
              - "desglose": dict por segmento con promedio %
              - "label": str — "Excelente" / "Bien" / "Sigue practicando"
              - "total_frames": int
              - "movement": str
        """
        if not self._frame_scores:
            return {"score_pct": 0.0, "label": "Sin datos", "desglose": {}, "total_frames": 0}

        # Promedio de scores por segmento
        desglose = {
            seg: round(np.mean(vals) * 100, 1) if vals else 0.0
            for seg, vals in self._segment_accumulators.items()
        }

        # Score global ponderado final
        score_total = sum(
            (desglose[seg] / 100.0) * SEGMENT_WEIGHTS[seg]
            for seg in BODY_SEGMENTS
        )
        score_pct = round(score_total * 100, 1)

        # Mejor y peor segmento
        mejor = max(desglose, key=desglose.get)
        peor  = min(desglose, key=desglose.get)

        result = {
            "movement": self._movement_name,
            "score_pct": score_pct,
            "label": score_label(score_total),
            "desglose": desglose,
            "mejor_segmento": mejor,
            "peor_segmento": peor,
            "total_frames": len(self._frame_scores),
        }

        self._session_active = False
        self._save_to_historial(result)
        return result

    def _save_to_historial(self, result: dict):
        """Guarda el resultado en el historial de los últimos 5 intentos."""
        os.makedirs(os.path.dirname(HISTORIAL_FILE), exist_ok=True)

        historial = []
        if os.path.exists(HISTORIAL_FILE):
            try:
                with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
                    historial = json.load(f)
            except Exception:
                historial = []

        entry = {
            "fecha": datetime.now().isoformat(),
            "movement": result["movement"],
            "score_pct": result["score_pct"],
            "label": result["label"],
            "desglose": result["desglose"],
        }
        historial.append(entry)
        # Mantener solo los últimos 5
        historial = historial[-5:]

        with open(HISTORIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(historial, f, indent=4, ensure_ascii=False)
        print(f"[ScoringEngine] Historial actualizado: {result['score_pct']}% — {result['label']}")

    def get_historial(self) -> list:
        """Retorna el historial de los últimos 5 intentos."""
        if not os.path.exists(HISTORIAL_FILE):
            return []
        try:
            with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    @property
    def current_frame_count(self) -> int:
        return len(self._frame_scores)

    @property
    def is_active(self) -> bool:
        return self._session_active


# ── Test rápido (ejecutar directamente) ────────────────────────────────────────
if __name__ == "__main__":
    import random

    print("=== Test ScoringEngine ===\n")
    engine = ScoringEngine()
    engine.start_session("secuencia_01")

    # Simular 60 frames con datos aleatorios
    for i in range(60):
        # Pose del estudiante (simulada)
        fake_student = {
            str(j): {
                'x': random.uniform(-0.5, 0.5),
                'y': random.uniform(-0.5, 0.5),
                'z': random.uniform(-0.1, 0.1),
                'v': 0.9
            } for j in range(33)
        }
        # Referencia del LSTM (simulada como array 99)
        fake_reference = np.random.randn(99).astype(np.float32)

        result = engine.evaluate_frame(fake_student, fake_reference)
        if i % 15 == 0:
            print(f"Frame {i:3d} | Overall: {result['overall']:.2f} | Color: {result['overall_color']}")
            for seg, score in result['segments'].items():
                print(f"         {seg:<20} {score:.2f}  {result['colors'][seg]}")
            print()

    final = engine.end_session()
    print("\n=== RESULTADO FINAL ===")
    print(f"Score: {final['score_pct']}% — {final['label']}")
    print(f"Total frames: {final['total_frames']}")
    print(f"Mejor segmento: {final['mejor_segmento']} ({final['desglose'][final['mejor_segmento']]}%)")
    print(f"Peor segmento:  {final['peor_segmento']} ({final['desglose'][final['peor_segmento']]}%)")
    print("\nDesglose por segmento:")
    for seg, pct in final['desglose'].items():
        bar = "█" * int(pct / 5)
        print(f"  {seg:<22} {pct:5.1f}%  {bar}")
