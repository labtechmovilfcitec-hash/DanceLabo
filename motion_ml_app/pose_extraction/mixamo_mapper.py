import numpy as np

class MixamoMapper:
    """
    Traduce los landmarks espaciales de MediaPipe a vectores direccionales
    normalizados para el rig de Mixamo en Unity.

    Incluye:
    - Restricción de eje Z para brazos (evita que vayan hacia atrás del cuerpo).
    - Freeze de última pose válida cuando la visibilidad de un landmark baja.
    """

    # Mapeo de huesos Mixamo a pares de landmarks MediaPipe (Padre -> Hijo)
    # MediaPipe IDs:
    # 11: left_shoulder, 13: left_elbow, 15: left_wrist
    # 12: right_shoulder, 14: right_elbow, 16: right_wrist
    # 23: left_hip, 25: left_knee, 27: left_ankle
    # 24: right_hip, 26: right_knee, 28: right_ankle
    BONE_PAIRS = {
        "mixamorig:LeftArm":      (11, 13),
        "mixamorig:RightArm":     (12, 14),
        "mixamorig:LeftForeArm":  (13, 15),
        "mixamorig:RightForeArm": (14, 16),
        "mixamorig:LeftUpLeg":    (23, 25),
        "mixamorig:RightUpLeg":   (24, 26),
        "mixamorig:LeftLeg":      (25, 27),
        "mixamorig:RightLeg":     (26, 28),
    }

    # Huesos que aplican restricción de eje Z (brazos/antebrazos solamente).
    # Los huesos de piernas NO se restringen — necesitan doblar hacia atrás.
    ARM_BONES = {
        "mixamorig:LeftArm",
        "mixamorig:RightArm",
        "mixamorig:LeftForeArm",
        "mixamorig:RightForeArm",
    }

    def __init__(self, enable_z_clamp=True, z_clamp_min=0.0, visibility_threshold=0.3, neck_z_multiplier=0.0):
        """
        Args:
            enable_z_clamp (bool): Activa la restricción de eje Z para brazos.
                                   Impide que los brazos vayan detrás del cuerpo.
            z_clamp_min (float):   Valor mínimo permitido para el componente Z de
                                   un vector de brazo. 0.0 = solo hacia adelante/lateral.
            un valor negativo pequeño (ej. -0.2) permite un poco
                                   de movimiento hacia atrás si se necesita.
            visibility_threshold (float): Confianza mínima de MediaPipe. Si baja de este
                                          umbral, se usa la última pose válida en lugar
                                          de descartar el dato.
            neck_z_multiplier (float): Factor de escala para el eje Z del cuello.
                                       Por defecto 0.0 (ignora profundidad ruidosa de MediaPipe).
        """
        self.enable_z_clamp = enable_z_clamp
        self.z_clamp_min = z_clamp_min
        self.visibility_threshold = visibility_threshold
        self.neck_z_multiplier = neck_z_multiplier

        # Diccionario que guarda el último vector válido por hueso.
        # Usado para "congelar" la pose cuando la visibilidad es baja.
        self._last_valid_vectors: dict = {}

    # ------------------------------------------------------------------
    # Coordenadas
    # ------------------------------------------------------------------

    def _mp_to_unity_coords(self, lm):
        """
        Convierte el sistema de coordenadas de MediaPipe al de Unity.
        MediaPipe: +X derecha, +Y abajo, +Z atrás (lejos de cámara).
        Unity:     +X derecha, +Y arriba, +Z adelante (hacia la cámara).
        """
        return np.array([lm['x'], -lm['y'], -lm['z']])

    # ------------------------------------------------------------------
    # Restricción de eje
    # ------------------------------------------------------------------

    def _apply_z_clamp(self, bone_name: str, direction: np.ndarray) -> np.ndarray:
        """
        Restringe el componente Z de un vector de brazo para que no vaya
        por debajo de z_clamp_min (por defecto 0 = no va hacia atrás).
        Después de clamping se re-normaliza para mantener el vector unitario.
        """
        if not self.enable_z_clamp or bone_name not in self.ARM_BONES:
            return direction

        z_original = direction[2]
        direction[2] = max(self.z_clamp_min, direction[2])

        # Si el clamping cambió algo, renormalizar
        if direction[2] != z_original:
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm

        return direction

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def get_bone_vectors(self, landmarks_data: dict) -> dict:
        """
        Recibe el diccionario de landmarks de MediaPipe y devuelve un diccionario
        con los vectores de dirección normalizados para cada hueso de Mixamo.

        Comportamiento ante baja visibilidad:
          - Si AMBOS landmarks son visibles → calcula vector fresco.
          - Si alguno tiene baja visibilidad → usa el último vector válido
            guardado en caché (_last_valid_vectors). Si no hay caché, omite el hueso.
        """
        vectors = {}

        for bone_name, (parent_id, child_id) in self.BONE_PAIRS.items():
            if parent_id not in landmarks_data or child_id not in landmarks_data:
                continue

            parent_lm = landmarks_data[parent_id]
            child_lm  = landmarks_data[child_id]

            low_visibility = (
                parent_lm.get('v', 1.0) < self.visibility_threshold or
                child_lm.get('v',  1.0) < self.visibility_threshold
            )

            if low_visibility:
                # Freeze: usar la última pose válida si existe
                if bone_name in self._last_valid_vectors:
                    vectors[bone_name] = self._last_valid_vectors[bone_name]
                # Si no hay caché, simplemente omitir (sin jitter)
                continue

            # Calcular vector fresco
            pos_parent = self._mp_to_unity_coords(parent_lm)
            pos_child  = self._mp_to_unity_coords(child_lm)

            direction = pos_child - pos_parent

            # Normalizar
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm

            # Aplicar restricción de eje Z (solo brazos)
            direction = self._apply_z_clamp(bone_name, direction)

            bone_vector = {
                "x": float(direction[0]),
                "y": float(direction[1]),
                "z": float(direction[2]),
            }

            vectors[bone_name] = bone_vector
            # Actualizar caché de última pose válida
            self._last_valid_vectors[bone_name] = bone_vector

        # ------------------------------------------------------------------
        # Caderas / Hips (mixamorig:Hips)
        # El hueso "hips" en Unity tiene T-Pose dir apuntando HACIA ARRIBA
        # (desde mixamorig:Hips hacia mixamorig:Spine), por lo que el vector
        # enviado debe ser el mismo que spine: desde el centro de caderas
        # hacia el centro de hombros.
        # NOTA: No usar cross-product (daba vector lateral que causaba flip de 90°).
        # ------------------------------------------------------------------
        hips_spine_ids = [11, 12, 23, 24]
        if all(idx in landmarks_data for idx in hips_spine_ids):
            hips_visible = all(
                landmarks_data[idx].get('v', 1.0) >= self.visibility_threshold
                for idx in hips_spine_ids
            )
            if hips_visible:
                shoulder_center = self._mp_to_unity_coords({
                    'x': (landmarks_data[11]['x'] + landmarks_data[12]['x']) / 2,
                    'y': (landmarks_data[11]['y'] + landmarks_data[12]['y']) / 2,
                    'z': (landmarks_data[11]['z'] + landmarks_data[12]['z']) / 2,
                })
                hip_center = self._mp_to_unity_coords({
                    'x': (landmarks_data[23]['x'] + landmarks_data[24]['x']) / 2,
                    'y': (landmarks_data[23]['y'] + landmarks_data[24]['y']) / 2,
                    'z': (landmarks_data[23]['z'] + landmarks_data[24]['z']) / 2,
                })
                hips_dir  = shoulder_center - hip_center
                hips_norm = np.linalg.norm(hips_dir)
                if hips_norm > 1e-6:
                    hips_dir = hips_dir / hips_norm
                    hips_vec = {
                        "x": float(hips_dir[0]),
                        "y": float(hips_dir[1]),
                        "z": float(hips_dir[2]),
                    }
                    vectors["mixamorig:Hips"] = hips_vec
                    self._last_valid_vectors["mixamorig:Hips"] = hips_vec
            elif "mixamorig:Hips" in self._last_valid_vectors:
                vectors["mixamorig:Hips"] = self._last_valid_vectors["mixamorig:Hips"]

        # ------------------------------------------------------------------
        # Espina dorsal inferior (mixamorig:Spine)
        # Vector desde el centro de las caderas al centro de los hombros
        # ------------------------------------------------------------------
        spine_ids = [11, 12, 23, 24]
        if all(idx in landmarks_data for idx in spine_ids):
            spine_visible = all(
                landmarks_data[idx].get('v', 1.0) >= self.visibility_threshold
                for idx in spine_ids
            )

            if spine_visible:
                shoulder_center = self._mp_to_unity_coords({
                    'x': (landmarks_data[11]['x'] + landmarks_data[12]['x']) / 2,
                    'y': (landmarks_data[11]['y'] + landmarks_data[12]['y']) / 2,
                    'z': (landmarks_data[11]['z'] + landmarks_data[12]['z']) / 2,
                })
                hip_center = self._mp_to_unity_coords({
                    'x': (landmarks_data[23]['x'] + landmarks_data[24]['x']) / 2,
                    'y': (landmarks_data[23]['y'] + landmarks_data[24]['y']) / 2,
                    'z': (landmarks_data[23]['z'] + landmarks_data[24]['z']) / 2,
                })

                spine_dir  = shoulder_center - hip_center
                spine_norm = np.linalg.norm(spine_dir)
                if spine_norm > 1e-6:
                    spine_dir = spine_dir / spine_norm
                    spine_vec = {
                        "x": float(spine_dir[0]),
                        "y": float(spine_dir[1]),
                        "z": float(spine_dir[2]),
                    }
                    # Spine y Spine1 usan el mismo vector (torso entero)
                    # Spine1 puede afinarse en el futuro si se tiene mas datos
                    vectors["mixamorig:Spine"]  = spine_vec
                    vectors["mixamorig:Spine1"] = spine_vec
                    self._last_valid_vectors["mixamorig:Spine"]  = spine_vec
                    self._last_valid_vectors["mixamorig:Spine1"] = spine_vec

            else:
                # Freeze espina si hay cache
                for key in ("mixamorig:Spine", "mixamorig:Spine1"):
                    if key in self._last_valid_vectors:
                        vectors[key] = self._last_valid_vectors[key]

        # ------------------------------------------------------------------
        # Cuello / Neck (mixamorig:Neck)
        # MediaPipe no tiene un landmark de cuello dedicado.
        # Se estima como vector desde el centro de hombros hacia la nariz (id 0).
        # NOTA: La profundidad (Z) de landmarks faciales es muy ruidosa en MediaPipe.
        # Multiplicar Z por un factor pequeno (neckZMultiplier) para suprimir el ruido.
        # self.neck_z_multiplier is defined in __init__ (defaults to 0.0)
        neck_ids = [0, 11, 12]
        if all(idx in landmarks_data for idx in neck_ids):
            neck_visible = all(
                landmarks_data[idx].get('v', 1.0) >= self.visibility_threshold
                for idx in neck_ids
            )
            if neck_visible:
                nose = self._mp_to_unity_coords(landmarks_data[0])
                sh_c = self._mp_to_unity_coords({
                    'x': (landmarks_data[11]['x'] + landmarks_data[12]['x']) / 2,
                    'y': (landmarks_data[11]['y'] + landmarks_data[12]['y']) / 2,
                    'z': (landmarks_data[11]['z'] + landmarks_data[12]['z']) / 2,
                })
                neck_dir  = nose - sh_c
                # Suprimir Z ruidoso: la profundidad de la nariz en MediaPipe es poco fiable
                neck_dir[2] *= self.neck_z_multiplier
                neck_norm = np.linalg.norm(neck_dir)
                if neck_norm > 1e-6:
                    neck_dir = neck_dir / neck_norm
                    neck_vec = {
                        "x": float(neck_dir[0]),
                        "y": float(neck_dir[1]),
                        "z": float(neck_dir[2]),
                    }
                    vectors["mixamorig:Neck"] = neck_vec
                    self._last_valid_vectors["mixamorig:Neck"] = neck_vec
            elif "mixamorig:Neck" in self._last_valid_vectors:
                vectors["mixamorig:Neck"] = self._last_valid_vectors["mixamorig:Neck"]

        return vectors

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def reset_cache(self):
        """Limpia el caché de poses válidas. Llamar al iniciar/detener una sesión."""
        self._last_valid_vectors.clear()
