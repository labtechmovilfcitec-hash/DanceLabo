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
        "mixamorig:LeftHand":     (15, 19),
        "mixamorig:RightHand":    (16, 20),
        "mixamorig:LeftUpLeg":    (23, 25),
        "mixamorig:RightUpLeg":   (24, 26),
        "mixamorig:LeftLeg":      (25, 27),
        "mixamorig:RightLeg":     (26, 28),
    }

    # Huesos que aplican restricción de eje Z (brazos, antebrazos y manos).
    # Los huesos de piernas NO se restringen — necesitan doblar hacia atrás.
    ARM_BONES = {
        "mixamorig:LeftArm",
        "mixamorig:RightArm",
        "mixamorig:LeftForeArm",
        "mixamorig:RightForeArm",
        "mixamorig:LeftHand",
        "mixamorig:RightHand",
    }

    def __init__(self, enable_z_clamp=False, z_clamp_min=-0.2, visibility_threshold=0.3, neck_z_multiplier=0.0):
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

        # Buffer circular para suavizado del body_yaw.
        # Promedio de sin/cos maneja correctamente el wrap-around en +/-180 grados.
        self._yaw_buffer_size = 7
        self._yaw_sin_history: list = []
        self._yaw_cos_history: list = []

        # Calibración de ancho 2D de hombros para aislar movimientos de brazos de giros del cuerpo
        self._max_shoulder_width_2d = 0.20

    # ------------------------------------------------------------------
    # Coordenadas
    # ------------------------------------------------------------------

    def _mp_to_unity_coords(self, lm):
        """
        Convierte el sistema de coordenadas de MediaPipe al de Unity.
        MediaPipe: +X derecha, +Y abajo, +Z atrás (lejos de cámara).
        Unity:     +X derecha, +Y arriba, +Z adelante (hacia la cámara).
        """
        # MediaPipe: +X derecha, +Y abajo, +Z alejándose de la cámara
        # Unity:     +X derecha, +Y arriba, +Z hacia la cámara
        # Solo invertimos Y. Z se deja sin negar para que el yaw y la profundidad sean correctos.
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

        # Detección de colapso o pérdida de tracking:
        # Si la persona se sale del encuadre o se acerca demasiado, los landmarks se agrupan en una bolita (< 0.05).
        # En ese caso, retenemos la última pose válida para no deformar o colapsar el avatar.
        if 11 in landmarks_data and 12 in landmarks_data:
            lsh = landmarks_data[11]
            rsh = landmarks_data[12]
            dist_2d = float(np.hypot(abs(lsh['x'] - rsh['x']), abs(lsh['y'] - rsh['y'])))
            if dist_2d < 0.05:
                return self._last_valid_vectors.copy() if self._last_valid_vectors else {}

        for bone_name, (parent_id, child_id) in self.BONE_PAIRS.items():
            if parent_id not in landmarks_data or child_id not in landmarks_data:
                continue

            parent_lm = landmarks_data[parent_id]
            child_lm  = landmarks_data[child_id]

            # Fallback inteligente para manos: si el dedo índice está tapado, usar meñique
            if bone_name == "mixamorig:LeftHand" and child_lm.get('v', 1.0) < self.visibility_threshold:
                if 17 in landmarks_data and landmarks_data[17].get('v', 1.0) >= self.visibility_threshold:
                    child_lm = landmarks_data[17]
            elif bone_name == "mixamorig:RightHand" and child_lm.get('v', 1.0) < self.visibility_threshold:
                if 18 in landmarks_data and landmarks_data[18].get('v', 1.0) >= self.visibility_threshold:
                    child_lm = landmarks_data[18]

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
        # Vector desde el centro de las caderas al centro de los hombros.
        # Robusto ante encuadres de escritorio: si las caderas están parcialmente
        # ocultas por el escritorio, se infiere su base para que la reverencia funcione.
        # ------------------------------------------------------------------
        if 11 in landmarks_data and 12 in landmarks_data:
            sh_v = (landmarks_data[11].get('v', 1.0) + landmarks_data[12].get('v', 1.0)) / 2.0
            if sh_v >= 0.20:
                shoulder_center = self._mp_to_unity_coords({
                    'x': (landmarks_data[11]['x'] + landmarks_data[12]['x']) / 2,
                    'y': (landmarks_data[11]['y'] + landmarks_data[12]['y']) / 2,
                    'z': (landmarks_data[11]['z'] + landmarks_data[12]['z']) / 2,
                })

                has_hips = (23 in landmarks_data and 24 in landmarks_data and
                            landmarks_data[23].get('v', 1.0) >= 0.10 and landmarks_data[24].get('v', 1.0) >= 0.10)

                if has_hips:
                    hip_center = self._mp_to_unity_coords({
                        'x': (landmarks_data[23]['x'] + landmarks_data[24]['x']) / 2,
                        'y': (landmarks_data[23]['y'] + landmarks_data[24]['y']) / 2,
                        'z': (landmarks_data[23]['z'] + landmarks_data[24]['z']) / 2,
                    })
                else:
                    # Cadera inferida anatómicamente bajo los hombros
                    hip_center = np.array([shoulder_center[0], shoulder_center[1] - 0.45, shoulder_center[2] - 0.10])

                spine_dir  = shoulder_center - hip_center
                spine_norm = np.linalg.norm(spine_dir)
                if spine_norm > 1e-6:
                    spine_dir = spine_dir / spine_norm
                    spine_vec = {
                        "x": float(spine_dir[0]),
                        "y": float(spine_dir[1]),
                        "z": float(spine_dir[2]),
                    }
                    vectors["mixamorig:Spine"]  = spine_vec
                    vectors["mixamorig:Spine1"] = spine_vec
                    self._last_valid_vectors["mixamorig:Spine"]  = spine_vec
                    self._last_valid_vectors["mixamorig:Spine1"] = spine_vec
            elif "mixamorig:Spine" in self._last_valid_vectors:
                vectors["mixamorig:Spine"]  = self._last_valid_vectors["mixamorig:Spine"]
                vectors["mixamorig:Spine1"] = self._last_valid_vectors.get("mixamorig:Spine1", self._last_valid_vectors["mixamorig:Spine"])

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

        # ------------------------------------------------------------------
        # Rotación de Cabeza (Head Rotation: Yaw, Pitch, Roll)
        # Permite que la cabeza y cuello roten de forma realista y expresiva.
        # ------------------------------------------------------------------
        head_rot = self._compute_head_rotation(landmarks_data)
        if head_rot is not None:
            vectors["head_rotation"] = head_rot
            self._last_valid_vectors["head_rotation"] = head_rot
        elif "head_rotation" in self._last_valid_vectors:
            vectors["head_rotation"] = self._last_valid_vectors["head_rotation"]

        # ------------------------------------------------------------------
        # Giro corporal (Body Yaw)
        # Enviado como "body_yaw" (float en grados) para que MixamoAnimator
        # lo aplique como rotación del GameObject raíz del avatar.
        # ------------------------------------------------------------------
        yaw = self._compute_body_yaw(landmarks_data)
        if yaw is not None:
            vectors["body_yaw"] = yaw
            self._last_valid_vectors["body_yaw"] = yaw
        elif "body_yaw" in self._last_valid_vectors:
            vectors["body_yaw"] = self._last_valid_vectors["body_yaw"]

        # ------------------------------------------------------------------
        # Posición Raíz / Desplazamiento (Root Position)
        # Permite que el personaje camine y salte siguiendo al usuario.
        # ------------------------------------------------------------------
        root_pos = self._compute_root_position(landmarks_data)
        if root_pos is not None:
            vectors["root_position"] = root_pos
            self._last_valid_vectors["root_position"] = root_pos
        elif "root_position" in self._last_valid_vectors:
            vectors["root_position"] = self._last_valid_vectors["root_position"]

        return vectors

    # ------------------------------------------------------------------
    # Rotación de cabeza (Head Rotation)
    # ------------------------------------------------------------------

    def _compute_head_rotation(self, landmarks_data: dict) -> dict | None:
        """
        Calcula la rotación 3D de la cabeza (Pitch, Yaw, Roll en grados)
        usando los landmarks faciales de MediaPipe (nariz 0, ojos 2/5).
        Asentir (Pitch) usa el desplazamiento vertical 2D para respuesta fiel y natural al decir 'SÍ'.
        Giro 360° suaviza la transición hacia espaldas sin deformaciones ni caídas de cabeza.
        """
        if not all(idx in landmarks_data for idx in [0, 2, 5]):
            return None

        nose = landmarks_data[0]
        l_eye = landmarks_data[2]
        r_eye = landmarks_data[5]

        # Si está de espaldas, la cabeza se mantiene alineada y neutra con el torso
        if landmarks_data.get('is_backward', False):
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        if nose.get('v', 1.0) < 0.20 or (l_eye.get('v', 1.0) < 0.20 and r_eye.get('v', 1.0) < 0.20):
            return None

        dx = float(l_eye['x'] - r_eye['x'])
        dy = -float(l_eye['y'] - r_eye['y']) # Invertir Y porque en imagen +Y va hacia abajo
        eye_dist = abs(dx)

        # Factor de confianza según distancia 2D entre ojos:
        # Al girar de perfil/espaldas, eye_dist disminuye. Desvanecemos a neutro suavemente
        # en lugar de dividir por valores cercanos a cero (que causaban la deformación/caída de cabeza).
        confidence = float(np.clip((eye_dist - 0.015) / 0.035, 0.0, 1.0))

        # 1. Roll (inclinación oreja hacia hombro): limitado anatómicamente a +/-30°
        roll_deg = float(np.clip(np.degrees(np.arctan2(dy, max(1e-4, dx))), -30.0, 30.0)) * confidence

        # 2. Yaw (girar cabeza a izquierda o derecha):
        mid_eyes_x = (l_eye['x'] + r_eye['x']) / 2.0
        yaw_ratio = (nose['x'] - mid_eyes_x) / max(1e-4, eye_dist * 0.5)
        yaw_deg = float(np.clip(yaw_ratio * 55.0, -55.0, 55.0)) * confidence

        # 3. Pitch (asentir con la cabeza - decir 'SÍ'):
        # Basado en la distancia vertical 2D de la nariz respecto a los ojos, normalizada por eye_dist.
        # Cuando asientes hacia abajo, la nariz baja respecto a los ojos (ratio sube).
        # Cuando miras hacia arriba, la nariz sube hacia los ojos (ratio baja).
        mid_eyes_y = (l_eye['y'] + r_eye['y']) / 2.0
        vert_offset = float(nose['y'] - mid_eyes_y) # Positivo porque nariz está debajo de ojos
        ratio = vert_offset / max(1e-4, eye_dist)
        pitch_delta = ratio - 0.65 # 0.65 es la proporción neutra promedio
        pitch_deg = float(np.clip(pitch_delta * 90.0, -35.0, 35.0)) * confidence

        return {
            "pitch": round(pitch_deg, 2),
            "yaw": round(yaw_deg, 2),
            "roll": round(roll_deg, 2),
        }

    # ------------------------------------------------------------------
    # Giro del cuerpo completo (Body Yaw)
    # ------------------------------------------------------------------

    def _compute_body_yaw(self, landmarks_data: dict) -> float | None:
        """
        Calcula el ángulo de rotación horizontal del cuerpo (yaw en grados de -180° a +180°).
        0° = de frente a la cámara, ±90° = perfil, ±180° = de espaldas.
        Inmune a movimientos de brazos (Macarena, Six Seven, baile) mientras el cuerpo está de frente.
        """
        if not all(idx in landmarks_data for idx in [11, 12]):
            return None

        lsh = landmarks_data[11]
        rsh = landmarks_data[12]

        if lsh.get('v', 1.0) < 0.15 or rsh.get('v', 1.0) < 0.15:
            return None

        lsh_u = self._mp_to_unity_coords(lsh)
        rsh_u = self._mp_to_unity_coords(rsh)

        # Distancia 2D entre hombros en el plano de la cámara
        dx_2d = float(np.hypot(lsh['x'] - rsh['x'], lsh['y'] - rsh['y']))
        if dx_2d > self._max_shoulder_width_2d:
            self._max_shoulder_width_2d = 0.95 * self._max_shoulder_width_2d + 0.05 * dx_2d

        # Vector hombro izquierdo → hombro derecho en el plano horizontal XZ
        shoulder_vec = lsh_u - rsh_u

        x_component = float(shoulder_vec[0])
        z_component = float(shoulder_vec[2]) * 0.65

        is_backward = landmarks_data.get('is_backward', False)

        # Si el detector de orientación confirma que la persona está de espaldas:
        # El vector pertenece al hemisferio posterior (x_component < 0)
        if is_backward:
            if x_component > 0:
                x_component = -x_component
            if abs(x_component) < 1e-4 and abs(z_component) < 1e-4:
                x_component = -1.0

        norm = np.hypot(x_component, z_component)
        if norm < 1e-6:
            return None

        yaw_rad = np.arctan2(z_component, x_component)
        raw_yaw_deg = float(np.degrees(yaw_rad))

        # Restricción geométrica:
        # Si la persona está de frente, el ancho 2D de hombros debe encogerse para que haya rotación real.
        # Al mover los brazos o bailar sin girar el torso, dx_2d se mantiene en ~100%,
        # impidiendo que la profundidad ruidosa de los brazos gire falsamente al personaje.
        if not is_backward and self._max_shoulder_width_2d > 0.05:
            width_ratio = min(1.0, dx_2d / max(1e-4, self._max_shoulder_width_2d * 0.90))
            max_yaw_allowed = float(np.degrees(np.arccos(width_ratio))) + 12.0
            raw_yaw_deg = float(np.clip(raw_yaw_deg, -max_yaw_allowed, max_yaw_allowed))

            # Zona muerta de 7° para absorber movimientos sutiles de brazos sin girar el cuerpo
            if abs(raw_yaw_deg) < 7.0:
                raw_yaw_deg = 0.0

        # Suavizado circular: maneja de forma continua y suave todo el rango de -180° a +180°
        yaw_rad_smoothed = np.radians(raw_yaw_deg)
        self._yaw_sin_history.append(float(np.sin(yaw_rad_smoothed)))
        self._yaw_cos_history.append(float(np.cos(yaw_rad_smoothed)))
        if len(self._yaw_sin_history) > self._yaw_buffer_size:
            self._yaw_sin_history.pop(0)
            self._yaw_cos_history.pop(0)

        avg_sin = float(np.mean(self._yaw_sin_history))
        avg_cos = float(np.mean(self._yaw_cos_history))
        return float(np.degrees(np.arctan2(avg_sin, avg_cos)))


    # ------------------------------------------------------------------
    # Desplazamiento Raíz (Root Position — caminar y saltar)
    # ------------------------------------------------------------------

    def _compute_root_position(self, landmarks_data: dict) -> dict | None:
        """
        Calcula la posición del torso del usuario para que el avatar
        camine, se desplace y salte replicando al usuario.
        Usa el centro de hombros como referencia estable (no salta si
        las caderas se ocultan al girar).
        """
        if not all(idx in landmarks_data for idx in [0, 11, 12]):
            return None

        lsh  = landmarks_data[11]
        rsh  = landmarks_data[12]
        nose = landmarks_data[0]

        if (lsh.get('v', 1.0) < 0.1 or
                rsh.get('v', 1.0) < 0.1 or
                nose.get('v', 1.0) < 0.1):
            return None

        lsh_u  = self._mp_to_unity_coords(lsh)
        rsh_u  = self._mp_to_unity_coords(rsh)
        nose_u = self._mp_to_unity_coords(nose)

        center = (lsh_u + rsh_u) / 2.0

        # Profundidad estimada como distancia vertical nariz→hombros
        # (invariante ante la rotación horizontal del usuario)
        torso_height = float(nose_u[1] - center[1]) * 4.5
        torso_height = max(0.05, torso_height)

        return {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(torso_height),
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def reset_cache(self):
        """Limpia el caché de poses válidas. Llamar al iniciar/detener una sesión."""
        self._last_valid_vectors.clear()
