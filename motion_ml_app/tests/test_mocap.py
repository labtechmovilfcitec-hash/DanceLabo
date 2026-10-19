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

if __name__ == '__main__':
    unittest.main()
