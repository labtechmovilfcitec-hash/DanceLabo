import numpy as np
import time

class OneEuroFilter:
    """
    Filtro adaptativo One Euro Filter (Casiez et al., CHI 2012).
    Elimina micro-vibraciones y temblores a baja velocidad manteniendo latencia cero en movimientos rápidos.
    """
    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        edx = self._alpha(self.d_cutoff, dt)
        dx_hat = edx * dx + (1.0 - edx) * self.dx_prev
        self.dx_prev = dx_hat

        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)

        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

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

    def __init__(self, enable_z_clamp=False, z_clamp_min=-0.2, visibility_threshold=0.20, neck_z_multiplier=0.0):
        """
        Args:
            enable_z_clamp (bool): Activa la restricción de eje Z para brazos.
                                   Impide que los brazos vayan detrás del cuerpo.
            z_clamp_min (float):   Valor mínimo permitido para el componente Z de
                                   un vector de brazo. 0.0 = solo hacia adelante/lateral.
            un valor negativo pequeño (ej. -0.2) permite un poco
                                   de movimiento hacia atrás si se necesita.
            visibility_threshold (float): Confianza mínima de MediaPipe. 0.20 permite
                                          fluidez en bailes donde los brazos cruzan el torso
                                          sin congelarse prematuramente.
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

        # Buffer circular para suavizado del body_yaw (5 frames ~ 0.08s para rotación instantánea).
        # Promedio de sin/cos maneja correctamente el wrap-around en +/-180 grados.
        self._yaw_buffer_size = 5
        self._yaw_sin_history: list = []
        self._yaw_cos_history: list = []
        self._last_stable_yaw: float = 0.0

        # Filtros adaptativos One Euro Filter por cada hueso para máxima fluidez orgánica
        self._filters: dict = {}

    def _filter_vector(self, bone_name: str, vec: np.ndarray) -> np.ndarray:
        """Aplica suavizado adaptativo One Euro Filter para eliminar micro-temblores sin retraso."""
        if bone_name not in self._filters:
            self._filters[bone_name] = OneEuroFilter(min_cutoff=1.5, beta=0.02)
        filtered = self._filters[bone_name].filter(vec, time.time())
        norm = np.linalg.norm(filtered)
        if norm > 1e-6:
            filtered = filtered / norm
        return filtered

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

        # Detección de colapso o pérdida real de tracking:
        # En vistas de perfil (de lado), los hombros se alinean en el eje de visión de la cámara (dist_2d < 0.03),
        # pero la persona sigue erguida (torso_height > 0.05).
        # Solo se considera colapso si AMBAS distancias (entre hombros y altura del torso) colapsan a un punto.
        if 11 in landmarks_data and 12 in landmarks_data:
            lsh = landmarks_data[11]
            rsh = landmarks_data[12]
            dist_2d = float(np.hypot(abs(lsh['x'] - rsh['x']), abs(lsh['y'] - rsh['y'])))
            torso_h = 1.0
            if 23 in landmarks_data and 24 in landmarks_data:
                torso_h = float(max(abs(lsh['y'] - landmarks_data[23]['y']), abs(rsh['y'] - landmarks_data[24]['y'])))
            if dist_2d < 0.012 and torso_h < 0.035:
                return self._last_valid_vectors.copy() if self._last_valid_vectors else {}

        for bone_name, (parent_id, child_id) in self.BONE_PAIRS.items():
            if parent_id not in landmarks_data or child_id not in landmarks_data:
                continue

            parent_lm = landmarks_data[parent_id]
            child_lm  = landmarks_data[child_id]

            # Fallback inteligente para manos: si el dedo índice está tapado, usar meñique o pulgar
            if bone_name == "mixamorig:LeftHand" and child_lm.get('v', 1.0) < self.visibility_threshold:
                if 17 in landmarks_data and landmarks_data[17].get('v', 1.0) >= self.visibility_threshold:
                    child_lm = landmarks_data[17]
                elif 21 in landmarks_data and landmarks_data[21].get('v', 1.0) >= self.visibility_threshold:
                    child_lm = landmarks_data[21]
            elif bone_name == "mixamorig:RightHand" and child_lm.get('v', 1.0) < self.visibility_threshold:
                if 18 in landmarks_data and landmarks_data[18].get('v', 1.0) >= self.visibility_threshold:
                    child_lm = landmarks_data[18]
                elif 22 in landmarks_data and landmarks_data[22].get('v', 1.0) >= self.visibility_threshold:
                    child_lm = landmarks_data[22]

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
            # Si world_landmarks (MediaPipe 3D en metros) está disponible, lo usamos para precisión anatómica
            # Esto resuelve completamente las distorsiones de profundidad 2D en vistas de perfil y giros
            wlms = landmarks_data.get('world_landmarks')
            if (wlms and parent_id in wlms and child_id in wlms and
                wlms[parent_id].get('v', 1.0) >= self.visibility_threshold and
                wlms[child_id].get('v', 1.0) >= self.visibility_threshold):
                p_w = wlms[parent_id]
                c_w = wlms[child_id]
                # MediaPipe world: +X derecha, +Y abajo, +Z alejándose
                # Unity:          +X derecha, +Y arriba, +Z hacia cámara (adelante)
                direction = np.array([
                    c_w['x'] - p_w['x'],
                    -(c_w['y'] - p_w['y']),
                    -(c_w['z'] - p_w['z'])
                ])
            else:
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

            # Suavizado adaptativo One Euro Filter (elimina jitter estático y da fluidez instantánea)
            direction = self._filter_vector(bone_name, direction)

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
                lsh_u = self._mp_to_unity_coords(landmarks_data[11])
                rsh_u = self._mp_to_unity_coords(landmarks_data[12])
                shoulder_center = (lsh_u + rsh_u) / 2.0

                has_hips = (23 in landmarks_data and 24 in landmarks_data and
                            landmarks_data[23].get('v', 1.0) >= 0.10 and landmarks_data[24].get('v', 1.0) >= 0.10)

                if has_hips:
                    hip_center = (self._mp_to_unity_coords(landmarks_data[23]) +
                                  self._mp_to_unity_coords(landmarks_data[24])) / 2.0
                else:
                    hip_center = np.array([shoulder_center[0], shoulder_center[1] - 0.45, shoulder_center[2] - 0.10])

                # 1. Componente lateral (Roll): medido directamente de la inclinación de hombros en el plano frontal.
                # Al inclinarte a los lados, la línea de hombros se desnivelará (un hombro sube y el otro baja).
                # La columna debe ser perpendicular a los hombros para inclinarse de forma fiel y visible:
                is_backward = landmarks_data.get('is_backward', False)
                u = (rsh_u - lsh_u) if is_backward else (lsh_u - rsh_u)
                raw_shoulder_roll = float(np.arctan2(u[1], max(1e-4, abs(u[0]))))
                # Limitar el roll anatómico a +/-42° para mantener estabilidad orgánica y evitar deformaciones
                shoulder_roll = float(np.clip(raw_shoulder_roll, -np.radians(42.0), np.radians(42.0)))

                x_roll = -float(np.sin(shoulder_roll))
                y_roll = float(np.cos(shoulder_roll))

                # 2. Componente de profundidad (Pitch): inclinación hacia adelante (reverencia/saludo)
                # Si world_landmarks está disponible, usamos la orientación 3D métrica en metros
                wlms = landmarks_data.get('world_landmarks')
                if wlms and has_hips and all(idx in wlms for idx in [11, 12, 23, 24]):
                    sh_c_w = np.array([
                        (wlms[11]['x'] + wlms[12]['x']) / 2.0,
                        (wlms[11]['y'] + wlms[12]['y']) / 2.0,
                        (wlms[11]['z'] + wlms[12]['z']) / 2.0
                    ])
                    hip_c_w = np.array([
                        (wlms[23]['x'] + wlms[24]['x']) / 2.0,
                        (wlms[23]['y'] + wlms[24]['y']) / 2.0,
                        (wlms[23]['z'] + wlms[24]['z']) / 2.0
                    ])
                    # En Unity: +Y arriba (-y), +Z adelante hacia cámara (-z)
                    w_spine = np.array([
                        sh_c_w[0] - hip_c_w[0],
                        -(sh_c_w[1] - hip_c_w[1]),
                        -(sh_c_w[2] - hip_c_w[2])
                    ])
                    w_norm = np.linalg.norm(w_spine)
                    if w_norm > 1e-6:
                        w_spine /= w_norm
                        pitch_rad = float(np.arctan2(w_spine[2], max(0.1, w_spine[1])))
                        # Reverencia permitida hasta 60° (1.05 rad), hiperextensión atrás limitada a -10° (-0.17 rad)
                        pitch_rad = float(np.clip(pitch_rad, -0.17, 1.05))
                        z_pitch = float(np.sin(pitch_rad))
                        y_pitch = float(np.cos(pitch_rad))
                    else:
                        z_pitch = 0.0
                        y_pitch = 1.0
                else:
                    if has_hips:
                        hip_center = (self._mp_to_unity_coords(landmarks_data[23]) +
                                      self._mp_to_unity_coords(landmarks_data[24])) / 2.0
                    else:
                        hip_center = np.array([shoulder_center[0], shoulder_center[1] - 0.45, shoulder_center[2] - 0.10])
                    z_pitch = float(shoulder_center[2] - hip_center[2])
                    tilt_factor = abs(x_roll)
                    if tilt_factor > 0.08:
                        z_pitch *= max(0.0, 1.0 - tilt_factor * 0.8)
                    z_pitch = max(-0.12, z_pitch)
                    y_pitch = 1.0

                spine_dir = np.array([x_roll, y_roll * y_pitch, z_pitch])
                spine_norm = np.linalg.norm(spine_dir)
                if spine_norm > 1e-6:
                    spine_dir = spine_dir / spine_norm
                    spine_dir = self._filter_vector("mixamorig:Spine", spine_dir)
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

        vectors["is_backward"] = bool(landmarks_data.get('is_backward', False))

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
        dy = -float(l_eye['y'] - r_eye['y']) # Invertir Y porque en imagen +Y va hacia abajo (sistema Cartesiano)
        eye_dist = float(np.hypot(dx, dy)) # Distancia euclidiana 2D invariante a la inclinación lateral
        if eye_dist < 1e-4:
            return None

        # Factor de confianza según distancia 2D entre ojos (invariante a inclinación lateral):
        # Al girar de perfil/espaldas, eye_dist disminuye por escorzo 3D.
        confidence = float(np.clip((eye_dist - 0.015) / 0.035, 0.0, 1.0))

        # 1. Roll (inclinación de ojos en el plano frontal de la cámara):
        roll_deg = float(np.clip(np.degrees(np.arctan2(dy, dx)), -45.0, 45.0)) * confidence

        # 2. Vectores unitarios en el sistema intrínseco de la cara:
        # u_hat apunta a lo largo del eje ocular (de ojo derecho a ojo izquierdo)
        u_hat = np.array([dx / eye_dist, dy / eye_dist])
        # v_hat apunta perpendicularmente hacia abajo (de los ojos hacia nariz y mentón en Cartesiano +Y arriba)
        v_hat = np.array([u_hat[1], -u_hat[0]])

        # Vector de la nariz respecto al punto medio ocular en coordenadas Cartesianas:
        mid_x = (l_eye['x'] + r_eye['x']) / 2.0
        mid_y = -((l_eye['y'] + r_eye['y']) / 2.0)
        nose_x = nose['x']
        nose_y = -nose['y']
        d_nose = np.array([nose_x - mid_x, nose_y - mid_y])

        # 3. Yaw (rotación horizontal de cabeza izquierda/derecha):
        # Medido como la proyección del desplazamiento nasal sobre el eje ocular u_hat.
        # Es 100% inmune a la inclinación lateral (roll) del cuerpo o la cabeza.
        d_yaw = float(np.dot(d_nose, u_hat))
        yaw_ratio = d_yaw / max(1e-4, eye_dist * 0.5)
        yaw_deg = float(np.clip(yaw_ratio * 55.0, -55.0, 55.0)) * confidence

        # 4. Pitch (asentir con la cabeza - decir 'SÍ'):
        # Medido como la proyección a lo largo del eje vertical v_hat de la cara.
        # Es 100% inmune a la inclinación lateral y no explota ni colapsa al ladearse.
        d_pitch = float(np.dot(d_nose, v_hat))
        pitch_ratio = d_pitch / max(1e-4, eye_dist)
        pitch_delta = pitch_ratio - 0.65 # 0.65 es la proporción neutra promedio
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
        Usa los landmarks 3D métricos (world_landmarks) cuando están disponibles y combina
        hombros (55%) y caderas (45%) para máxima estabilidad ante movimientos de brazos y piernas
        (Macarena, bailes, paso de jamal / cat dance).
        Des-proyecta la inclinación lateral (roll) del torso en el plano frontal para que
        inclinarse a los lados no genere falsos giros de yaw.
        Alinea el signo con Unity (+X derecha = +90°, extendiendo brazos al frente de forma natural).
        """
        if not all(idx in landmarks_data for idx in [11, 12]):
            return None

        lsh_raw = landmarks_data[11]
        rsh_raw = landmarks_data[12]
        if lsh_raw.get('v', 1.0) < 0.15 or rsh_raw.get('v', 1.0) < 0.15:
            return None

        # Preferir world_landmarks (métrica 3D en metros) si están disponibles para exactitud anatómica
        wlms = landmarks_data.get('world_landmarks')
        if (wlms and 11 in wlms and 12 in wlms and 23 in wlms and 24 in wlms and
            wlms[11].get('v', 1.0) >= 0.15 and wlms[12].get('v', 1.0) >= 0.15):
            # MediaPipe world: +X derecha, +Y abajo, +Z alejándose de la cámara
            # Unity:          +X derecha, +Y arriba, +Z hacia la cámara
            lsh_u = np.array([wlms[11]['x'], -wlms[11]['y'], -wlms[11]['z']])
            rsh_u = np.array([wlms[12]['x'], -wlms[12]['y'], -wlms[12]['z']])
            lhip_u = np.array([wlms[23]['x'], -wlms[23]['y'], -wlms[23]['z']])
            rhip_u = np.array([wlms[24]['x'], -wlms[24]['y'], -wlms[24]['z']])
            has_hips = True
        else:
            lsh_u = self._mp_to_unity_coords(lsh_raw)
            rsh_u = self._mp_to_unity_coords(rsh_raw)
            has_hips = (23 in landmarks_data and 24 in landmarks_data and
                        landmarks_data[23].get('v', 1.0) >= 0.10 and landmarks_data[24].get('v', 1.0) >= 0.10)
            if has_hips:
                lhip_u = self._mp_to_unity_coords(landmarks_data[23])
                rhip_u = self._mp_to_unity_coords(landmarks_data[24])

        # Vector de ancho del torso: combinar hombros (55%) y caderas (45%)
        # Esto inmuniza el giro corporal contra movimientos de brazos (Macarena, cat dance)
        u_sh = lsh_u - rsh_u
        if has_hips:
            u_hip = lhip_u - rhip_u
            u = 0.55 * u_sh + 0.45 * u_hip
            v = ((lsh_u + rsh_u) / 2.0) - ((lhip_u + rhip_u) / 2.0)
        else:
            u = u_sh
            v = np.array([-u[1], u[0], 0.0])

        # Des-proyectar inclinación lateral (roll) del plano del torso:
        # Al inclinarse o agacharse a los lados, el vector de columna v se inclina en XY.
        # Des-rotar u y v en el plano XY cancela el roll y evita que la inclinación lateral contamine el yaw.
        v_norm_xy = float(np.hypot(v[0], v[1]))
        if v_norm_xy > 1e-4:
            roll_angle = float(np.arctan2(v[0], v[1]))
            cos_r = float(np.cos(-roll_angle))
            sin_r = float(np.sin(-roll_angle))
            u_unrolled = np.array([
                cos_r * u[0] - sin_r * u[1],
                sin_r * u[0] + cos_r * u[1],
                u[2]
            ])
            v_unrolled = np.array([
                cos_r * v[0] - sin_r * v[1],
                sin_r * v[0] + cos_r * v[1],
                v[2]
            ])
        else:
            u_unrolled = u
            v_unrolled = v

        # Normal al plano del torso: n = u × v (apunta hacia adelante saliendo del pecho)
        n = np.cross(u_unrolled, v_unrolled)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            return None
        n = n / norm_n

        is_backward = landmarks_data.get('is_backward', False)

        # En Unity: +Z es adelante (hacia la cámara), +X es derecha
        # n[0] es X, n[2] es Z
        nx = float(n[0])
        nz = float(n[2])

        if abs(nx) < 0.50:
            if is_backward and nz > 0:
                nz = -nz
            elif not is_backward and nz < -0.70:
                nz = abs(nz)

        # Ángulo yaw en el plano horizontal XZ:
        # En Unity, rotar alrededor de +Y por +90° orienta el avatar hacia +X (pantalla-derecha).
        # Por tanto, np.arctan2(nx, nz) alinea exactamente el avatar con el usuario en pantalla.
        raw_yaw_deg = float(np.degrees(np.arctan2(nx, nz)))
        # Zona muerta frontal sutil de 5° para absorber micro-ruido en reposo sin bloquear giros reales
        if not is_backward and abs(raw_yaw_deg) < 5.0:
            raw_yaw_deg = 0.0

        # Suavizado circular continuo de -180° a +180°
        yaw_rad_smoothed = np.radians(raw_yaw_deg)
        self._yaw_sin_history.append(float(np.sin(yaw_rad_smoothed)))
        self._yaw_cos_history.append(float(np.cos(yaw_rad_smoothed)))
        if len(self._yaw_sin_history) > self._yaw_buffer_size:
            self._yaw_sin_history.pop(0)
            self._yaw_cos_history.pop(0)

        avg_sin = float(np.mean(self._yaw_sin_history))
        avg_cos = float(np.mean(self._yaw_cos_history))
        self._last_stable_yaw = float(np.degrees(np.arctan2(avg_sin, avg_cos)))
        return self._last_stable_yaw


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
        """Limpia el caché de poses válidas e historiales de yaw y filtros. Llamar al iniciar/detener una sesión."""
        self._last_valid_vectors.clear()
        self._yaw_sin_history.clear()
        self._yaw_cos_history.clear()
        self._last_stable_yaw = 0.0
        for f in self._filters.values():
            f.reset()
        self._filters.clear()
