"""
scoring_engine.py — D-06 / E-inv: Motor de Puntuación con Huesos Mixamo
Dance Labo · Sprint 3

Compara la pose del estudiante (huesos Mixamo desde Unity via UDP)
vs la pose de referencia generada por MotionLSTMGenerator, frame a frame.
Feature vector: 9 huesos × 3 coordenadas = 27 valores por frame.
"""

import numpy as np
import json
import os
from datetime import datetime


# ── Huesos usados (en el mismo orden que dataset_builder.py) ─────────────────
EXPECTED_BONES = [
    "mixamorig:LeftArm",
    "mixamorig:RightArm",
    "mixamorig:LeftForeArm",
    "mixamorig:RightForeArm",
    "mixamorig:LeftUpLeg",
    "mixamorig:RightUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:RightLeg",
    "mixamorig:Spine",
]
BONE_INDEX = {bone: i for i, bone in enumerate(EXPECTED_BONES)}
FEATURE_SIZE = len(EXPECTED_BONES) * 3  # 27

# ── Segmentos corporales agrupados por hueso ──────────────────────────────────
# Cada valor es una lista de índices en EXPECTED_BONES
BODY_SEGMENTS = {
    "brazo_izquierdo":  ["mixamorig:LeftArm",   "mixamorig:LeftForeArm"],
    "brazo_derecho":    ["mixamorig:RightArm",   "mixamorig:RightForeArm"],
    "pierna_izquierda": ["mixamorig:LeftUpLeg",  "mixamorig:LeftLeg"],
    "pierna_derecha":   ["mixamorig:RightUpLeg", "mixamorig:RightLeg"],
    "torso":            ["mixamorig:Spine"],
}

# Peso de cada segmento en el score global (suma = 1.0)
SEGMENT_WEIGHTS = {
    "brazo_izquierdo":  0.25,
    "brazo_derecho":    0.25,
    "pierna_izquierda": 0.20,
    "pierna_derecha":   0.20,
    "torso":            0.10,
}

# Umbrales de evaluación
THRESHOLD_VERDE    = 0.85   # >= 85% → Verde (bien)
THRESHOLD_AMARILLO = 0.60   # >= 60% → Amarillo (regular)
# < 60% → Rojo (mal)

HISTORIAL_FILE = "data/historial_scores.json"


# ── Funciones de conversión ───────────────────────────────────────────────────

def bones_dict_to_vector(bones_dict: dict) -> np.ndarray:
    """
    Convierte el dict de huesos recibido desde Unity:
      {"mixamorig:LeftArm": {"x": .., "y": .., "z": ..}, ...}
    a un vector plano de 27 valores (orden fijo = EXPECTED_BONES).
    Huesos ausentes se llenan con ceros.
    """
    vector = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for bone, idx in BONE_INDEX.items():
        bdata = bones_dict.get(bone)
        if bdata:
            vector[idx * 3]     = bdata.get('x', 0.0)
            vector[idx * 3 + 1] = bdata.get('y', 0.0)
            vector[idx * 3 + 2] = bdata.get('z', 0.0)
    return vector


def lstm_frame_to_vector(frame_tensor) -> np.ndarray:
    """
    Convierte el frame de salida del LSTM (tensor o array de 27 dims) a np.array.
    """
    if hasattr(frame_tensor, 'detach'):
        return frame_tensor.detach().cpu().numpy().flatten()
    return np.array(frame_tensor, dtype=np.float32).flatten()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similitud coseno entre dos vectores → valor en [0, 1]."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float((np.dot(v1, v2) / (n1 * n2) + 1.0) / 2.0)


def segment_similarity(student_vec: np.ndarray,
                        reference_vec: np.ndarray,
                        bone_names: list):
    """
    Similitud coseno de un segmento corporal.
    Devuelve None si ambos vectores son cero (hueso no grabado en el dataset).
    """
    indices = [BONE_INDEX[b] for b in bone_names if b in BONE_INDEX]
    seg_s = np.concatenate([student_vec[i*3:(i+1)*3]   for i in indices])
    seg_r = np.concatenate([reference_vec[i*3:(i+1)*3] for i in indices])

    # Si ambos son cero, el hueso no estaba en los datos de entrenamiento → ignorar
    if np.linalg.norm(seg_s) < 1e-6 and np.linalg.norm(seg_r) < 1e-6:
        return None

    return cosine_similarity(seg_s, seg_r)


def score_label(score: float) -> str:
    if score >= THRESHOLD_VERDE:
        return "Excelente"
    elif score >= THRESHOLD_AMARILLO:
        return "Bien"
    return "Sigue practicando"


def color_for_score(score: float) -> str:
    if score >= THRESHOLD_VERDE:
        return "verde"
    elif score >= THRESHOLD_AMARILLO:
        return "amarillo"
    return "rojo"


# ── Clase principal ───────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Motor de puntuación acumulada — Dance Labo (D-06 / E-inv).

    Uso típico:
        engine = ScoringEngine()
        engine.start_session("macarena")

        for frame in frames:
            bones_from_unity = {"mixamorig:LeftArm": {"x":..,"y":..,"z":..}, ...}
            ref_frame_tensor  = lstm_model_output[frame_idx]   # (27,)
            result = engine.evaluate_frame(bones_from_unity, ref_frame_tensor)
            # result["overall"]  → 0.0–1.0
            # result["segments"] → {"brazo_izquierdo": 0.9, ...}
            # result["colors"]   → {"brazo_izquierdo": "verde", ...}

        final = engine.end_session()
        # final["score_pct"]   → 74.3
        # final["label"]       → "Bien"
        # final["desglose"]    → dict por segmento
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._frame_scores          = []
        self._segment_accumulators  = {seg: [] for seg in BODY_SEGMENTS}
        self._session_active        = False
        self._movement_name         = ""
        self._start_time            = None

    def start_session(self, movement_name: str):
        self.reset()
        self._session_active = True
        self._movement_name  = movement_name
        self._start_time     = datetime.now()
        print(f"[ScoringEngine] Sesión iniciada: {movement_name}")

    def evaluate_frame(self, bones_dict: dict, reference_frame_tensor) -> dict:
        """
        Evalúa un frame comparando la pose del estudiante vs referencia del LSTM.

        Args:
            bones_dict: dict {"mixamorig:LeftArm": {"x":.., "y":.., "z":..}, ...}
            reference_frame_tensor: tensor/array (27,) del LSTM

        Returns:
            dict con:
              overall        → float [0,1]
              segments       → dict por segmento, float [0,1]
              colors         → dict por segmento, str
              overall_color  → str
        """
        if not self._session_active:
            return {}

        student_vec   = bones_dict_to_vector(bones_dict)
        reference_vec = lstm_frame_to_vector(reference_frame_tensor)

        # Calcular similitud por segmento (None = hueso sin datos)
        segment_scores = {}
        valid_weight_total = 0.0
        for seg_name, bones in BODY_SEGMENTS.items():
            sim = segment_similarity(student_vec, reference_vec, bones)
            segment_scores[seg_name] = sim
            if sim is not None:
                valid_weight_total += SEGMENT_WEIGHTS[seg_name]
            self._segment_accumulators[seg_name].append(sim if sim is not None else None)

        # Score global — solo sobre segmentos con datos reales, normalizado
        overall = 0.0
        if valid_weight_total > 0:
            for seg_name, sim in segment_scores.items():
                if sim is not None:
                    overall += sim * (SEGMENT_WEIGHTS[seg_name] / valid_weight_total)

        colors = {
            seg: color_for_score(s) if s is not None else "gris"
            for seg, s in segment_scores.items()
        }

        result = {
            "overall":       overall,
            "segments":      segment_scores,
            "colors":        colors,
            "overall_color": color_for_score(overall),
        }
        self._frame_scores.append(result)
        return result

    def end_session(self) -> dict:
        """Finaliza la sesión y retorna resultado acumulado."""
        if not self._frame_scores:
            return {"score_pct": 0.0, "label": "Sin datos",
                    "desglose": {}, "total_frames": 0}

        # Promediar solo valores válidos por segmento (ignorar None)
        desglose = {}
        valid_segs = {}
        for seg, vals in self._segment_accumulators.items():
            real_vals = [v for v in vals if v is not None]
            if real_vals:
                pct = round(np.mean(real_vals) * 100, 1)
                desglose[seg] = pct
                valid_segs[seg] = pct
            else:
                desglose[seg] = None   # sin datos

        # Score global normalizado sobre segmentos con datos
        valid_weight_total = sum(
            SEGMENT_WEIGHTS[seg] for seg in valid_segs
        )
        score_total = 0.0
        if valid_weight_total > 0:
            for seg, pct in valid_segs.items():
                score_total += (pct / 100.0) * (SEGMENT_WEIGHTS[seg] / valid_weight_total)
        score_pct = round(score_total * 100, 1)

        mejor = max(valid_segs, key=valid_segs.get) if valid_segs else "N/A"
        peor  = min(valid_segs, key=valid_segs.get) if valid_segs else "N/A"

        result = {
            "movement":        self._movement_name,
            "score_pct":       score_pct,
            "label":           score_label(score_total),
            "desglose":        desglose,
            "mejor_segmento":  mejor,
            "peor_segmento":   peor,
            "total_frames":    len(self._frame_scores),
        }

        self._session_active = False
        self._save_to_historial(result)
        return result

    def _save_to_historial(self, result: dict):
        os.makedirs(os.path.dirname(HISTORIAL_FILE), exist_ok=True)
        historial = []
        if os.path.exists(HISTORIAL_FILE):
            try:
                with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
                    historial = json.load(f)
            except Exception:
                historial = []

        historial.append({
            "fecha":     datetime.now().isoformat(),
            "movement":  result["movement"],
            "score_pct": result["score_pct"],
            "label":     result["label"],
            "desglose":  result["desglose"],
        })
        historial = historial[-5:]

        with open(HISTORIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(historial, f, indent=4, ensure_ascii=False)
        print(f"[ScoringEngine] {result['score_pct']}% — {result['label']}")

    def get_historial(self) -> list:
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


# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    print("=== Test ScoringEngine (Mixamo 27 features) ===\n")

    engine = ScoringEngine()
    engine.start_session("macarena")

    for i in range(60):
        fake_bones = {
            bone: {'x': random.uniform(-1, 1),
                   'y': random.uniform(-1, 1),
                   'z': random.uniform(-1, 1)}
            for bone in EXPECTED_BONES
        }
        fake_ref = np.random.randn(27).astype(np.float32)
        result = engine.evaluate_frame(fake_bones, fake_ref)

        if i % 20 == 0:
            print(f"Frame {i:3d} | Overall: {result['overall']:.2f} | {result['overall_color']}")
            for seg, s in result['segments'].items():
                print(f"         {seg:<22} {s:.2f}  {result['colors'][seg]}")
            print()

    final = engine.end_session()
    print(f"\n=== RESULTADO FINAL ===")
    print(f"Score: {final['score_pct']}% — {final['label']}")
    for seg, pct in final['desglose'].items():
        bar = "█" * int(pct / 5)
        print(f"  {seg:<22} {pct:5.1f}%  {bar}")
