import os
import json
import torch
from ml.model import MotionLSTMGenerator

# Ruta por defecto alineada con train.py
DEFAULT_MODEL_PATH    = "data/motion_model.pt"
DEFAULT_LABEL_PATH    = "data/label_map.pt"
DEFAULT_SEQUENCES_DIR = "data/sequences"


class MotionPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_map = {}
        self.max_seq_length = 0
        self.is_loaded = False

        self.load_model(model_path)

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"No se encontró el modelo en '{path}'. Entrena la IA primero.")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Soporte para checkpoints que guardan label_map y max_seq_length,
            # y para los que solo guardan state_dict (compatibilidad hacia atrás).
            if isinstance(checkpoint, dict) and 'label_map' in checkpoint:
                self.label_map      = checkpoint['label_map']
                self.max_seq_length = checkpoint.get('max_seq_length', 150)
                state_dict          = checkpoint['model_state_dict']
            else:
                # Checkpoint antiguo: intentar cargar label_map por separado
                label_path = path.replace('motion_model.pt', 'label_map.pt')
                if os.path.exists(label_path):
                    self.label_map = torch.load(label_path, map_location='cpu')
                self.max_seq_length = 150
                state_dict = checkpoint

            num_classes = len(self.label_map)
            self.model  = MotionLSTMGenerator(
                num_classes=num_classes,
                max_seq_length=self.max_seq_length
            ).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            print(f"Modelo cargado — {num_classes} movimientos, "
                  f"seq_length={self.max_seq_length}")
            return True

        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False

    # ------------------------------------------------------------------
    # Modo 1: LSTM generativo
    # ------------------------------------------------------------------

    def predict_sequence(self, movement_name):
        """
        🧠 MODO LSTM — Genera la secuencia con el modelo entrenado.
        El resultado es una aproximación aprendida del movimiento (~95% similitud).
        Útil para variaciones e interpolaciones suaves.
        Retorna lista de dicts {bone: {x,y,z}}.
        """
        if not self.is_loaded:
            print("Error: modelo no cargado.")
            return []

        if movement_name not in self.label_map:
            # Fallback: quitar sufijo numérico (ej. Prueba2_1 -> Prueba2)
            import re
            cleaned_name = re.sub(r'_\d+$', '', movement_name)
            if cleaned_name in self.label_map:
                movement_name = cleaned_name
            else:
                print(f"[LSTM] Movimiento '{movement_name}' no reconocido. "
                      f"Disponibles: {list(self.label_map.keys())}")
                return []

        expected_bones = [
            "mixamorig:LeftArm",      "mixamorig:RightArm",
            "mixamorig:LeftForeArm",  "mixamorig:RightForeArm",
            "mixamorig:LeftUpLeg",    "mixamorig:RightUpLeg",
            "mixamorig:LeftLeg",      "mixamorig:RightLeg",
            "mixamorig:Spine",
        ]

        class_id = self.label_map[movement_name]

        with torch.no_grad():
            x = torch.tensor([class_id], dtype=torch.long).to(self.device)
            generated_seq = self.model(x, seq_length=self.max_seq_length)
            generated_seq = generated_seq.squeeze(0).cpu().numpy()  # (T, 27)

        frames_data = []
        for frame_features in generated_seq:
            bone_vectors = {}
            for i, bone in enumerate(expected_bones):
                base = i * 3
                bone_vectors[bone] = {
                    "x": float(frame_features[base]),
                    "y": float(frame_features[base + 1]),
                    "z": float(frame_features[base + 2]),
                }
            frames_data.append(bone_vectors)

        print(f"[LSTM] '{movement_name}' generado — {len(frames_data)} frames.")
        return frames_data

    # ------------------------------------------------------------------
    # Modo 2: Reproducción directa desde JSON
    # ------------------------------------------------------------------

    def playback_raw_sequence(self, movement_name,
                              sequences_dir=DEFAULT_SEQUENCES_DIR):
        """
        📼 MODO DIRECTO — Lee el JSON grabado y devuelve los frames exactos.
        No usa el modelo LSTM. Replica el movimiento frame por frame tal
        como fue capturado por la cámara. Es el modo más fiel a la grabación.

        Si hay múltiples archivos con el mismo movement_name (varias tomas),
        usa el más reciente por fecha de modificación.

        Retorna lista de dicts {bone: {x,y,z}}, o [] si no se encuentra.
        """
        if not os.path.exists(sequences_dir):
            print(f"[Directo] Carpeta no encontrada: '{sequences_dir}'")
            return []

        # Intentar limpiar sufijo numérico (ej. Prueba2_1 -> Prueba2)
        import re
        movement_name = re.sub(r'_\d+$', '', movement_name)

        # Buscar todos los JSON con el movement_name correcto
        matches = []
        for fname in os.listdir(sequences_dir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(sequences_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("movement_name") == movement_name:
                    matches.append((fpath, data))
            except Exception as e:
                print(f"[Directo] Error leyendo {fname}: {e}")

        if not matches:
            print(f"[Directo] No se encontró secuencia para '{movement_name}'.")
            return []

        # Si hay varias tomas, usar la más reciente
        matches.sort(key=lambda t: os.path.getmtime(t[0]), reverse=True)
        chosen_path, chosen_data = matches[0]
        raw_frames = chosen_data.get("frames", [])

        if len(matches) > 1:
            print(f"[Directo] {len(matches)} tomas de '{movement_name}'. "
                  f"Usando la más reciente: {os.path.basename(chosen_path)}")
        else:
            print(f"[Directo] '{movement_name}' — "
                  f"{len(raw_frames)} frames exactos desde "
                  f"{os.path.basename(chosen_path)}")

        # Extraer bone_vectors de cada frame del JSON
        frames_data = []
        for frame in raw_frames:
            landmarks = frame.get("landmarks", {})
            if landmarks:
                frames_data.append(landmarks)

        return frames_data
