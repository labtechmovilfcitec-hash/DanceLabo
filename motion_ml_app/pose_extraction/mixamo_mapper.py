import numpy as np

class MixamoMapper:
    """
    Traduce los landmarks espaciales de MediaPipe a vectores direccionales
    normalizados para el rig de Mixamo en Unity.
    """
    
    # Mapeo de huesos Mixamo a pares de landmarks MediaPipe (Padre -> Hijo)
    # MediaPipe IDs:
    # 11: left_shoulder, 13: left_elbow, 15: left_wrist
    # 12: right_shoulder, 14: right_elbow, 16: right_wrist
    # 23: left_hip, 25: left_knee, 27: left_ankle
    # 24: right_hip, 26: right_knee, 28: right_ankle
    BONE_PAIRS = {
        "mixamorig:LeftArm": (11, 13),
        "mixamorig:RightArm": (12, 14),
        "mixamorig:LeftForeArm": (13, 15),
        "mixamorig:RightForeArm": (14, 16),
        "mixamorig:LeftUpLeg": (23, 25),
        "mixamorig:RightUpLeg": (24, 26),
        "mixamorig:LeftLeg": (25, 27),
        "mixamorig:RightLeg": (26, 28)
    }

    def __init__(self):
        pass

    def _mp_to_unity_coords(self, lm):
        """
        Convierte el sistema de coordenadas de MediaPipe al de Unity.
        MediaPipe: +X derecha, +Y abajo, +Z atrás (lejos de cámara).
        Unity: +X derecha, +Y arriba, +Z adelante (hacia la cámara / profundidad).
        """
        # Multiplicamos Y y Z por -1 para adaptar ejes
        return np.array([lm['x'], -lm['y'], -lm['z']])

    def get_bone_vectors(self, landmarks_data):
        """
        Recibe el diccionario de landmarks de MediaPipe y devuelve un diccionario
        con los vectores de dirección normalizados para cada hueso de Mixamo.
        """
        vectors = {}
        
        for bone_name, (parent_id, child_id) in self.BONE_PAIRS.items():
            if parent_id in landmarks_data and child_id in landmarks_data:
                parent_lm = landmarks_data[parent_id]
                child_lm = landmarks_data[child_id]
                
                # Omitir si la visibilidad es muy baja
                if parent_lm.get('v', 1.0) < 0.3 or child_lm.get('v', 1.0) < 0.3:
                    continue
                
                # Convertir a numpy arrays en coordenadas de Unity
                pos_parent = self._mp_to_unity_coords(parent_lm)
                pos_child = self._mp_to_unity_coords(child_lm)
                
                # Vector direccion: Destino - Origen
                direction = pos_child - pos_parent
                
                # Normalizar vector
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    
                vectors[bone_name] = {
                    "x": float(direction[0]),
                    "y": float(direction[1]),
                    "z": float(direction[2])
                }
                
        # Calcular un vector aproximado para la espina dorsal (mixamorig:Spine)
        # Vector desde el centro de las caderas al centro de los hombros
        if all(idx in landmarks_data for idx in [11, 12, 23, 24]):
            shoulder_center = self._mp_to_unity_coords({
                'x': (landmarks_data[11]['x'] + landmarks_data[12]['x']) / 2,
                'y': (landmarks_data[11]['y'] + landmarks_data[12]['y']) / 2,
                'z': (landmarks_data[11]['z'] + landmarks_data[12]['z']) / 2
            })
            hip_center = self._mp_to_unity_coords({
                'x': (landmarks_data[23]['x'] + landmarks_data[24]['x']) / 2,
                'y': (landmarks_data[23]['y'] + landmarks_data[24]['y']) / 2,
                'z': (landmarks_data[23]['z'] + landmarks_data[24]['z']) / 2
            })
            
            spine_dir = shoulder_center - hip_center
            spine_norm = np.linalg.norm(spine_dir)
            if spine_norm > 0:
                spine_dir = spine_dir / spine_norm
                vectors["mixamorig:Spine"] = {
                    "x": float(spine_dir[0]),
                    "y": float(spine_dir[1]),
                    "z": float(spine_dir[2])
                }
                
        return vectors
