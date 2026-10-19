import unittest
import numpy as np
from motion_ml_app.pose_extraction.mixamo_mapper import MixamoMapper

class TestMocapFixes(unittest.TestCase):
    def setUp(self):
        self.mapper = MixamoMapper()

    def test_lateral_leaning_yaw_immunity(self):
        """
        Verifica que inclinarse lateralmente (roll de torso de 15° a 75°)
        NO genere rotación horizontal (yaw) falsa.
        """
        base_lsh = np.array([0.20, 0.45, 0.0])
        base_rsh = np.array([-0.20, 0.45, 0.0])
        base_lhip = np.array([0.10, 0.0, 0.0])
        base_rhip = np.array([-0.10, 0.0, 0.0])

        for lean_deg in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
            rad = np.radians(lean_deg)
            c, s = np.cos(rad), np.sin(rad)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            lsh = R @ base_lsh
            rsh = R @ base_rsh
            lhip = R @ base_lhip
            rhip = R @ base_rhip

            lm = {
                11: {'x': lsh[0] + 0.5, 'y': -lsh[1] + 0.5, 'z': -lsh[2], 'v': 0.99},
                12: {'x': rsh[0] + 0.5, 'y': -rsh[1] + 0.5, 'z': -rsh[2], 'v': 0.99},
                23: {'x': lhip[0] + 0.5, 'y': -lhip[1] + 0.5, 'z': -lhip[2], 'v': 0.99},
                24: {'x': rhip[0] + 0.5, 'y': -rhip[1] + 0.5, 'z': -rhip[2], 'v': 0.99},
                'is_backward': False
            }

            self.mapper.reset_cache()
            yaw = self.mapper._compute_body_yaw(lm)
            self.assertIsNotNone(yaw, f"Yaw fue None para inclinación de {lean_deg}°")
            self.assertLess(abs(yaw), 8.0, 
                f"Inclinación lateral de {lean_deg}° causó giro falso de {yaw:.1f}° (debe ser < 8°)")

    def test_spine_lateral_lean_direction(self):
        """
        Verifica que al inclinarse a los lados, mixamorig:Spine capture la inclinación lateral (eje X)
        en la dirección correcta.
        """
        base_lsh = np.array([0.20, 0.45, 0.0])
        base_rsh = np.array([-0.20, 0.45, 0.0])
        base_hip = np.array([0.0, 0.0, 0.0])

        # Inclinación hacia la izquierda de la pantalla (+30° rota de +Y a -X)
        rad = np.radians(30)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        lsh = R @ base_lsh
        rsh = R @ base_rsh
        sh_c = (lsh + rsh) / 2.0

        lm = {
            11: {'x': lsh[0] + 0.5, 'y': -lsh[1] + 0.5, 'z': 0.0, 'v': 0.99},
            12: {'x': rsh[0] + 0.5, 'y': -rsh[1] + 0.5, 'z': 0.0, 'v': 0.99},
            23: {'x': 0.10 + 0.5, 'y': 0.5, 'z': 0.0, 'v': 0.99},
            24: {'x': -0.10 + 0.5, 'y': 0.5, 'z': 0.0, 'v': 0.99},
        }

        vectors = self.mapper.get_bone_vectors(lm)
        self.assertIn("mixamorig:Spine", vectors)
        spine = vectors["mixamorig:Spine"]
        # El vector debe estar inclinado a la izquierda (x < 0)
        self.assertLess(spine["x"], -0.15, "La columna debe reflejar la inclinación hacia la izquierda")

    def test_360_turn_smoothness(self):
        """
        Simula un giro continuo de 360° y verifica que no existan saltos bruscos (> 45°) entre frames sucesivos.
        """
        base_lsh = np.array([0.20, 0.45, 0.0])
        base_rsh = np.array([-0.20, 0.45, 0.0])
        base_hip = np.array([0.0, 0.0, 0.0])

        yaws = []
        self.mapper.reset_cache()

        # Giro continuo de 0° a 360° en pasos de 5°
        for deg in range(0, 360, 5):
            rad = np.radians(deg)
            # Rotación alrededor del eje vertical Y
            Ry = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
            lsh = Ry @ base_lsh
            rsh = Ry @ base_rsh

            is_backward = 90 < deg < 270

            lm = {
                11: {'x': lsh[0] + 0.5, 'y': -lsh[1] + 0.5, 'z': -lsh[2], 'v': 0.99},
                12: {'x': rsh[0] + 0.5, 'y': -rsh[1] + 0.5, 'z': -rsh[2], 'v': 0.99},
                23: {'x': 0.5, 'y': 0.5, 'z': 0.0, 'v': 0.99},
                24: {'x': 0.5, 'y': 0.5, 'z': 0.0, 'v': 0.99},
                'is_backward': is_backward
            }

            yaw = self.mapper._compute_body_yaw(lm)
            yaws.append(yaw)

        # Verificar que la trayectoria es suave y no tiene discontinuidades
        for i in range(1, len(yaws)):
            diff = abs(yaws[i] - yaws[i-1])
            if diff > 180: # Cruce de wrap ±180°
                diff = 360 - diff
            self.assertLess(diff, 35.0, f"Salto brusco de {diff:.1f}° entre frames en giro 360°")

    def test_macarena_arm_poses(self):
        """
        Verifica que posturas típicas de la Macarena (brazos al frente, cruzados al pecho, manos a orejas)
        generen vectores válidos, finitos y normalizados.
        """
        # Pose 1: Brazos al frente
        lm_front = {
            11: {'x': 0.6, 'y': 0.4, 'z': 0.0, 'v': 0.95}, # L shoulder
            12: {'x': 0.4, 'y': 0.4, 'z': 0.0, 'v': 0.95}, # R shoulder
            13: {'x': 0.6, 'y': 0.4, 'z': -0.3, 'v': 0.95}, # L elbow forward
            14: {'x': 0.4, 'y': 0.4, 'z': -0.3, 'v': 0.95}, # R elbow forward
            15: {'x': 0.6, 'y': 0.4, 'z': -0.6, 'v': 0.95}, # L wrist forward
            16: {'x': 0.4, 'y': 0.4, 'z': -0.6, 'v': 0.95}, # R wrist forward
            19: {'x': 0.6, 'y': 0.4, 'z': -0.7, 'v': 0.95},
            20: {'x': 0.4, 'y': 0.4, 'z': -0.7, 'v': 0.95},
        }

        vectors = self.mapper.get_bone_vectors(lm_front)
        for bone in ["mixamorig:LeftArm", "mixamorig:RightArm", "mixamorig:LeftForeArm", "mixamorig:RightForeArm"]:
            self.assertIn(bone, vectors)
            v = vectors[bone]
            norm = np.hypot(v['x'], np.hypot(v['y'], v['z']))
            self.assertAlmostEqual(norm, 1.0, places=3, msg=f"El vector de {bone} debe estar normalizado")
            self.assertFalse(np.isnan(v['x']) or np.isnan(v['y']) or np.isnan(v['z']))

    def test_profile_transition_no_jump(self):
        """
        Verifica que en la vista de perfil (~90°), si el detector facial
        parpadea entre is_backward True y False, el yaw no sufra un salto brusco (> 20°).
        """
        base_lsh = np.array([0.0, 0.45, -0.20])
        base_rsh = np.array([0.0, 0.45, 0.20])
        base_hip = np.array([0.0, 0.0, 0.0])

        lm_forward = {
            11: {'x': base_lsh[0] + 0.5, 'y': -base_lsh[1] + 0.5, 'z': -base_lsh[2], 'v': 0.99},
            12: {'x': base_rsh[0] + 0.5, 'y': -base_rsh[1] + 0.5, 'z': -base_rsh[2], 'v': 0.99},
            23: {'x': 0.5, 'y': 0.5, 'z': 0.0, 'v': 0.99},
            24: {'x': 0.5, 'y': 0.5, 'z': 0.0, 'v': 0.99},
            'is_backward': False
        }
        lm_backward = dict(lm_forward)
        lm_backward['is_backward'] = True

        self.mapper.reset_cache()
        yaw_f = self.mapper._compute_body_yaw(lm_forward)

        self.mapper.reset_cache()
        yaw_b = self.mapper._compute_body_yaw(lm_backward)

        self.assertIsNotNone(yaw_f)
        self.assertIsNotNone(yaw_b)
        diff = abs(yaw_f - yaw_b)
        self.assertLess(diff, 10.0, f"En vista de perfil el cambio de is_backward causó un salto de {diff:.1f}°")

    def test_is_backward_transmission(self):
        """
        Verifica que el estado is_backward se transmita fielmente a Unity
        para alternar automáticamente entre Mirror Mode (frente) y Direct Mode (espaldas).
        """
        lm = {
            11: {'x': 0.6, 'y': 0.4, 'z': 0.0, 'v': 0.95},
            12: {'x': 0.4, 'y': 0.4, 'z': 0.0, 'v': 0.95},
            'is_backward': True
        }
        vecs = self.mapper.get_bone_vectors(lm)
        self.assertIn("is_backward", vecs)
        self.assertTrue(vecs["is_backward"])

        lm['is_backward'] = False
        vecs = self.mapper.get_bone_vectors(lm)
        self.assertFalse(vecs["is_backward"])

    def test_head_rotation_tilt_invariance(self):
        """
        Verifica que al inclinar la cabeza a los lados (roll de 0° a 45°):
        1. eye_dist no colapse artificialmente (usando distancia euclidiana).
        2. El pitch (asentir) y el yaw (girar) permanezcan cercanos a 0° sin explotar ni fallar.
        """
        for angle in [-40, -30, -15, 0, 15, 30, 40]:
            rad = np.radians(angle)
            c, s = np.cos(rad), np.sin(rad)
            # Rotación rígida estándar en coordenadas Cartesianas (+X derecha, +Y arriba):
            # l_eye en (+0.025*c, +0.025*s), r_eye en (-0.025*c, -0.025*s)
            # nose en (+0.0325*s, -0.0325*c)
            # En imagen MediaPipe: x_img = 0.5 + x_cart, y_img = 0.4 - y_cart
            r_eye = {'x': 0.5 - 0.025 * c, 'y': 0.4 + 0.025 * s, 'v': 0.99}
            l_eye = {'x': 0.5 + 0.025 * c, 'y': 0.4 - 0.025 * s, 'v': 0.99}
            nose  = {'x': 0.5 + 0.0325 * s, 'y': 0.4 + 0.0325 * c, 'v': 0.99}

            lm = {0: nose, 2: l_eye, 5: r_eye, 'is_backward': False}
            hr = self.mapper._compute_head_rotation(lm)
            self.assertIsNotNone(hr, f"Head rotation fue None para ángulo de {angle}°")
            # El pitch debe mantenerse cerca de 0° (sin falsos cabeceos al inclinarse)
            self.assertLess(abs(hr['pitch']), 5.0,
                f"Al inclinarse {angle}°, pitch se distorsionó a {hr['pitch']}° (debe ser < 5°)")
            # El yaw debe mantenerse cerca de 0° (sin falsos giros de cabeza al inclinarse)
            self.assertLess(abs(hr['yaw']), 5.0,
                f"Al inclinarse {angle}°, yaw se distorsionó a {hr['yaw']}° (debe ser < 5°)")
            # El roll debe reflejar el ángulo inclinado
            self.assertAlmostEqual(hr['roll'], angle, delta=3.0,
                msg=f"Roll calculado ({hr['roll']}°) debe coincidir con ángulo inclinado ({angle}°)")

    def test_spine_lateral_lean_clamping_and_z_attenuation(self):
        """
        Verifica que al inclinarse a los lados:
        1. El roll se limite anatómicamente para evitar quiebres de columna.
        2. El ruido en Z sea atenuado para no generar colapsos hacia atrás.
        """
        # Postura de hombros con gran desnivel (inclinación lateral pronunciada) y ruido en Z
        lm = {
            11: {'x': 0.65, 'y': 0.55, 'z': -0.15, 'v': 0.99}, # hombro izquierdo abajo
            12: {'x': 0.35, 'y': 0.35, 'z': 0.15, 'v': 0.99},  # hombro derecho arriba
            23: {'x': 0.55, 'y': 0.80, 'z': 0.0, 'v': 0.99},
            24: {'x': 0.45, 'y': 0.80, 'z': 0.0, 'v': 0.99},
            'is_backward': False
        }
        vecs = self.mapper.get_bone_vectors(lm)
        self.assertIn("mixamorig:Spine", vecs)
        spine = vecs["mixamorig:Spine"]
        # Debe inclinarse a la derecha (x > 0)
        self.assertGreater(spine["x"], 0.20)
        # El componente Z debe estar atenuado y acotado (no menor a -0.12)
        self.assertGreaterEqual(spine["z"], -0.12)

    def test_pose_extractor_backward_hysteresis(self):
        """
        Verifica que la histéresis de orientación en PoseExtractor:
        1. No cambie a espaldas por una o dos caídas aisladas de detección.
        2. Cambie a espaldas tras 8 frames continuos de espalda (front_votes <= 2).
        3. No vuelva al frente por un falso positivo aislado.
        4. Vuelva al frente de forma limpia tras 5 frames confirmados.
        """
        from motion_ml_app.pose_extraction.landmark_extractor import PoseExtractor
        extractor = PoseExtractor()

        # Iniciar con 10 frames de frente
        for _ in range(10):
            extractor._face_detected_history.append(True)
        extractor._is_facing_backward = False

        # 1-2 frames sin detección (parpadeo o gesto rápido) NO deben cambiar a espaldas
        for _ in range(2):
            extractor._face_detected_history.pop(0)
            extractor._face_detected_history.append(False)
            front_votes = sum(1 for f in extractor._face_detected_history if f)
            if front_votes <= 2 and len(extractor._face_detected_history) >= 8:
                extractor._is_facing_backward = True
        self.assertFalse(extractor._is_facing_backward, "2 frames sin detección no deben activar espaldas")

        # 8 frames continuos sin rostro (usuario de espaldas) SÍ deben activar espaldas
        for _ in range(8):
            extractor._face_detected_history.pop(0)
            extractor._face_detected_history.append(False)
            front_votes = sum(1 for f in extractor._face_detected_history if f)
            if front_votes <= 2 and len(extractor._face_detected_history) >= 8:
                extractor._is_facing_backward = True
        self.assertTrue(extractor._is_facing_backward, "8 frames continuos sin rostro deben activar espaldas")

        # 1 frame con detección aislada (reflejo o ruido) NO debe desactivar espaldas
        extractor._face_detected_history.pop(0)
        extractor._face_detected_history.append(True)
        front_votes = sum(1 for f in extractor._face_detected_history if f)
        if extractor._is_facing_backward and front_votes >= 5:
            extractor._is_facing_backward = False
        self.assertTrue(extractor._is_facing_backward, "1 frame aislado no debe desactivar espaldas")

        # 5 frames con detección (usuario volteándose al frente) SÍ deben desactivar espaldas
        for _ in range(5):
            extractor._face_detected_history.pop(0)
            extractor._face_detected_history.append(True)
            front_votes = sum(1 for f in extractor._face_detected_history if f)
            if extractor._is_facing_backward and front_votes >= 5:
                extractor._is_facing_backward = False
        self.assertFalse(extractor._is_facing_backward, "5 frames con rostro deben restaurar el modo frente")

    def test_continuous_360_spin_tracking(self):
        """
        Verifica que una rotación completa de 360° no tenga rebotes ni zonas muertas al cruzar los 90° o 270°.
        """
        for deg in range(0, 360, 15):
            rad = np.radians(deg)
            # En MediaPipe (+X derecha, +Z alejándose de la cámara):
            # Rotar a la derecha (+yaw) desplaza hombro izquierdo atrás (+Z) y hombro derecho al frente (-Z)
            lsh = {'x': 0.5 + 0.2 * np.cos(rad), 'y': 0.3, 'z': +0.2 * np.sin(rad), 'v': 0.99}
            rsh = {'x': 0.5 - 0.2 * np.cos(rad), 'y': 0.3, 'z': -0.2 * np.sin(rad), 'v': 0.99}
            lhip = {'x': 0.5 + 0.1 * np.cos(rad), 'y': 0.7, 'z': +0.1 * np.sin(rad), 'v': 0.99}
            rhip = {'x': 0.5 - 0.1 * np.cos(rad), 'y': 0.7, 'z': -0.1 * np.sin(rad), 'v': 0.99}
            is_back = (120 <= deg <= 240)
            lm = {11: lsh, 12: rsh, 23: lhip, 24: rhip, 'is_backward': is_back}
            self.mapper.reset_cache()
            yaw = self.mapper._compute_body_yaw(lm)
            expected_yaw = deg if deg <= 180 else deg - 360
            diff = abs(yaw - expected_yaw)
            if diff > 180:
                diff = 360 - diff
            self.assertLess(diff, 15.0, f"Error excesivo en ángulo {deg}°: yaw={yaw:.1f} vs expected={expected_yaw:.1f}")

    def test_cat_dance_jamal_step_yaw_stability(self):
        """
        Verifica que al hacer el paso de jamal / baile del gatito
        (levantar mano y rodilla derecha en frente del pecho mirando a la cámara):
        1. El yaw corporal NO salte a 70°-90° (falso giro).
        2. Permanezca acotado en la zona frontal (< 28°).
        """
        # Cara mirando de frente a la cámara
        nose = {'x': 0.50, 'y': 0.30, 'z': 0.0, 'v': 0.99}
        l_eye = {'x': 0.52, 'y': 0.28, 'z': 0.0, 'v': 0.99}
        r_eye = {'x': 0.48, 'y': 0.28, 'z': 0.0, 'v': 0.99}
        l_ear = {'x': 0.55, 'y': 0.29, 'z': 0.0, 'v': 0.99}
        r_ear = {'x': 0.45, 'y': 0.29, 'z': 0.0, 'v': 0.99}

        # Simulación de escorzo/oclusión por levantar mano y rodilla en frente:
        # Hombros con desnivel y aparente desplazamiento en Z
        lsh = {'x': 0.58, 'y': 0.40, 'z': 0.0, 'v': 0.99}
        rsh = {'x': 0.46, 'y': 0.45, 'z': 0.15, 'v': 0.99} # rsh desplazado en Z por mano al frente
        lhip = {'x': 0.55, 'y': 0.70, 'z': 0.0, 'v': 0.99}
        rhip = {'x': 0.47, 'y': 0.65, 'z': 0.10, 'v': 0.99} # rhip desplazado por rodilla alzada

        wlms = {
            11: {'x': 0.18, 'y': -0.42, 'z': -0.19, 'v': 0.99},
            12: {'x': -0.16, 'y': -0.40, 'z': -0.10, 'v': 0.99}, # leve desplazamiento en Z por mano alzada
            23: {'x': 0.10, 'y': 0.0, 'z': 0.0, 'v': 0.99},
            24: {'x': -0.10, 'y': 0.0, 'z': 0.0, 'v': 0.99},
        }

        lm = {
            0: nose, 2: l_eye, 5: r_eye, 7: l_ear, 8: r_ear,
            11: lsh, 12: rsh, 23: lhip, 24: rhip,
            'world_landmarks': wlms,
            'is_backward': False
        }

        self.mapper.reset_cache()
        yaw = self.mapper._compute_body_yaw(lm)
        self.assertIsNotNone(yaw)
        self.assertLessEqual(abs(yaw), 28.0,
            f"El paso de jamal / cat dance no debe disparar falso yaw (obtenido: {yaw:.1f}°, debe ser <= 28°)")

    def test_profile_facing_arms_reach_forward(self):
        """
        Verifica que al estar de perfil mirando hacia la derecha (+X):
        1. El yaw sea positivo (+60° a +90°).
        2. Extender los brazos hacia la derecha (+X en espacio de mundo/cámara)
           se proyecte en Z POSITIVO (al frente del pecho) en el espacio local del avatar.
        """
        # Hombros alineados en perfil mirando a la derecha
        lsh = {'x': 0.50, 'y': 0.40, 'z': +0.18, 'v': 0.99}
        rsh = {'x': 0.50, 'y': 0.40, 'z': -0.18, 'v': 0.99}
        lhip = {'x': 0.50, 'y': 0.75, 'z': +0.10, 'v': 0.99}
        rhip = {'x': 0.50, 'y': 0.75, 'z': -0.10, 'v': 0.99}

        lm = {
            11: lsh, 12: rsh, 23: lhip, 24: rhip,
            'is_backward': False
        }

        self.mapper.reset_cache()
        yaw = self.mapper._compute_body_yaw(lm)
        self.assertIsNotNone(yaw)
        self.assertGreater(yaw, 60.0, f"De perfil mirando a la derecha, el yaw debe ser positivo (obtenido: {yaw:.1f}°)")

        # En Unity, localDir = Quaternion.Euler(0, -yaw, 0) * targetDir
        # Con targetDir hacia la derecha (+1, 0, 0):
        rad = np.radians(yaw)
        target_dir = np.array([1.0, 0.0, 0.0])
        # Rotación por -yaw en Y:
        # local_z = -(-sin(-yaw)) * target_dir[0] + cos(-yaw) * target_dir[2]
        # En Unity: z' = -sin(-rad) * x + cos(-rad) * z = sin(rad) * x + cos(rad) * z
        local_z = np.sin(rad) * target_dir[0]
        self.assertGreater(local_z, 0.70,
            f"Extender los brazos al frente en perfil debe dar Z local positivo (> 0.70, obtenido: {local_z:.2f})")

    def test_profile_shoulder_alignment_not_collapsed(self):
        """
        Verifica que cuando el usuario se pone de perfil (hombros alineados en línea de visión,
        distancia 2D < 0.025 pero torso erguido con altura > 0.05):
        1. NO se confunda con un colapso de tracking.
        2. NO congele los vectores en la pose anterior.
        3. Calcule vectores frescos para los brazos extendidos.
        """
        # Hombros superpuestos en 2D (perfil hacia la izquierda de la cámara)
        lm = {
            11: {'x': 0.50, 'y': 0.30, 'z': -0.15, 'v': 0.99},
            12: {'x': 0.51, 'y': 0.30, 'z': +0.15, 'v': 0.99},
            13: {'x': 0.35, 'y': 0.35, 'z': -0.15, 'v': 0.99}, # Codo izquierdo apuntando al frente
            15: {'x': 0.20, 'y': 0.35, 'z': -0.15, 'v': 0.99}, # Muñeca izquierda apuntando al frente
            23: {'x': 0.50, 'y': 0.65, 'z': -0.05, 'v': 0.99},
            24: {'x': 0.51, 'y': 0.65, 'z': +0.05, 'v': 0.99},
            'is_backward': False
        }
        self.mapper.reset_cache()
        vecs = self.mapper.get_bone_vectors(lm)
        self.assertIn("mixamorig:LeftArm", vecs, "Brazo izquierdo debe calcularse en postura de perfil")
        self.assertIn("mixamorig:LeftForeArm", vecs, "Antebrazo izquierdo debe calcularse en postura de perfil")
        # El antebrazo debe apuntar en X negativa (hacia donde apunta el usuario)
        self.assertLess(vecs["mixamorig:LeftForeArm"]["x"], -0.80)

    def test_forward_reverencia_world_landmarks(self):
        """
        Verifica que al hacer una reverencia hacia la cámara (pitch hacia adelante ~50°):
        1. MixamoMapper use world_landmarks para calcular la flexión métrica 3D completa.
        2. El componente Z de mixamorig:Spine sea claramente positivo (> 0.65) inclinando al personaje adelante.
        """
        # Inclinación en 3D métrico (hombros desplazados hacia la cámara en Z respecto a caderas)
        rad = np.radians(50.0) # 50 grados adelante
        c, s = float(np.cos(rad)), float(np.sin(rad)) # c ~ 0.64 (arriba), s ~ 0.77 (adelante)

        wlms = {
            11: {'x': 0.15, 'y': -0.45 * c, 'z': -0.45 * s, 'v': 0.99},
            12: {'x': -0.15, 'y': -0.45 * c, 'z': -0.45 * s, 'v': 0.99},
            23: {'x': 0.10, 'y': 0.0, 'z': 0.0, 'v': 0.99},
            24: {'x': -0.10, 'y': 0.0, 'z': 0.0, 'v': 0.99},
        }
        lm = {
            11: {'x': 0.55, 'y': 0.30, 'z': -0.10, 'v': 0.99},
            12: {'x': 0.45, 'y': 0.30, 'z': -0.10, 'v': 0.99},
            23: {'x': 0.53, 'y': 0.60, 'z': 0.0, 'v': 0.99},
            24: {'x': 0.47, 'y': 0.60, 'z': 0.0, 'v': 0.99},
            'world_landmarks': wlms,
            'is_backward': False
        }
        self.mapper.reset_cache()
        vecs = self.mapper.get_bone_vectors(lm)
        self.assertIn("mixamorig:Spine", vecs)
        spine = vecs["mixamorig:Spine"]
        # Debe tener Z positivo pronunciado (inclinación frontal en Unity)
        self.assertGreater(spine["z"], 0.65, f"Spine Z ({spine['z']:.3f}) debe ser > 0.65 para una reverencia de 50°")
        # El ángulo debe ser cercano a 50°
        pitch_angle = np.degrees(np.arctan2(spine["z"], spine["y"]))
        self.assertAlmostEqual(pitch_angle, 50.0, delta=5.0)

if __name__ == '__main__':
    unittest.main()

