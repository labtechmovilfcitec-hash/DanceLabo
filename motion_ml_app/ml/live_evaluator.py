"""
ml/live_evaluator.py — E-04: Evaluación en tiempo real de la pose del estudiante.

Carga la secuencia LSTM de un movimiento y la compara frame a frame con los
vectores de huesos que llegan de la cámara, actualizando el score en vivo.

Uso típico desde main_window.py:
    evaluator = LiveEvaluator("Prueba1")
    if evaluator.is_ready:
        result = evaluator.evaluate(bone_vectors_dict)
        # result["overall"]       → 0.0 – 1.0
        # result["overall_color"] → "verde" | "amarillo" | "rojo"
        # result["segments"]      → {"brazo_izquierdo": 0.9, ...}
        # result["colors"]        → {"brazo_izquierdo": "verde", ...}
"""

from ml.predictor import MotionPredictor
from ml.scoring_engine import ScoringEngine, bones_dict_to_vector


class LiveEvaluator:
    """
    Evaluador en tiempo real para una sesión de danza.

    1. Carga la secuencia LSTM generada para *movement_name*.
    2. Convierte cada frame a un vector numpy de 27 dimensiones (9 huesos × 3 coords).
    3. Por cada frame de cámara, compara con el frame de referencia correspondiente
       y devuelve el resultado completo del ScoringEngine.
    4. Cuando llega al último frame de referencia, reinicia el loop automáticamente.
    """

    def __init__(self, movement_name: str,
                 model_path: str = "data/motion_model.pt"):
        self.movement_name      = movement_name
        self.reference_vectors  = []   # list of np.ndarray (27,)
        self.frame_idx          = 0
        self.total_frames       = 0
        self.engine             = ScoringEngine()
        self._load_reference(movement_name, model_path)

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def _load_reference(self, name: str, model_path: str):
        """Genera la secuencia de referencia con el LSTM y la convierte a vectores."""
        predictor = MotionPredictor(model_path)
        frames_dicts = predictor.predict_sequence(name)

        if not frames_dicts:
            print(f"[LiveEval] ⚠ Movimiento '{name}' no encontrado en el modelo. "
                  "Verifica que está entrenado.")
            return

        # Convertir cada dict de huesos → vector numpy de 27 valores
        self.reference_vectors = [bones_dict_to_vector(f) for f in frames_dicts]
        self.total_frames = len(self.reference_vectors)
        self.engine.start_session(name)
        print(f"[LiveEval] ✅ '{name}' — {self.total_frames} frames de referencia cargados.")

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True si la referencia fue cargada correctamente."""
        return self.total_frames > 0

    def evaluate(self, student_bones: dict) -> dict | None:
        """
        Evalúa un frame del estudiante contra el frame de referencia actual.

        Args:
            student_bones: dict con formato {boneName: {"x": float, "y": float, "z": float}}
                           tal como lo devuelve MixamoMapper.get_bone_vectors().

        Returns:
            dict con claves:
                overall        → float [0, 1]
                overall_color  → "verde" | "amarillo" | "rojo"
                segments       → {segmento: float | None}
                colors         → {segmento: str}
            o None si no está listo.
        """
        if not self.is_ready or not student_bones:
            return None

        ref_vec = self.reference_vectors[self.frame_idx]
        result  = self.engine.evaluate_frame(student_bones, ref_vec)

        # Avanzar al siguiente frame (loop circular)
        self.frame_idx = (self.frame_idx + 1) % self.total_frames

        return result

    def get_progress_pct(self) -> float:
        """Porcentaje de avance en la secuencia de referencia (0.0 – 1.0)."""
        if self.total_frames == 0:
            return 0.0
        return self.frame_idx / self.total_frames

    def reset(self):
        """Reinicia la evaluación desde el frame 0."""
        self.frame_idx = 0
        if self.movement_name:
            self.engine.start_session(self.movement_name)
        print(f"[LiveEval] Reiniciado '{self.movement_name}'.")
