import os
import torch
from ml.model import MotionLSTMGenerator

class MotionPredictor:
    def __init__(self, model_path="data/models/motion_lstm.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_map = {}
        self.max_seq_length = 0
        self.is_loaded = False
        
        self.load_model(model_path)
        
    def load_model(self, path):
        if not os.path.exists(path):
            print(f"No se encontro el modelo en {path}. Entrena la IA primero.")
            return False
            
        try:
            # Cargar el diccionario guardado en trainer.py
            checkpoint = torch.load(path, map_location=self.device)
            self.label_map = checkpoint['label_map']
            self.max_seq_length = checkpoint['max_seq_length']
            
            num_classes = len(self.label_map)
            self.model = MotionLSTMGenerator(num_classes=num_classes, max_seq_length=self.max_seq_length).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo inferencia
            self.is_loaded = True
            print("Modelo cargado exitosamente.")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False

    def predict_sequence(self, movement_name):
        """
        Recibe el nombre del baile y genera la secuencia completa de vectores de Mixamo.
        Retorna una lista de diccionarios (frames), similar a lo que recibe udp_server.send_pose.
        """
        if not self.is_loaded or movement_name not in self.label_map:
            print(f"Error: Movimiento '{movement_name}' no reconocido o modelo no cargado.")
            return []
            
        # Orden de huesos esperado por dataset_builder
        expected_bones = [
            "mixamorig:LeftArm", "mixamorig:RightArm",
            "mixamorig:LeftForeArm", "mixamorig:RightForeArm",
            "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg",
            "mixamorig:LeftLeg", "mixamorig:RightLeg",
            "mixamorig:Spine"
        ]
            
        class_id = self.label_map[movement_name]
        
        with torch.no_grad():
            x = torch.tensor([class_id], dtype=torch.long).to(self.device)
            # Generar secuencia, output shape: (1, max_seq_length, 27)
            generated_seq = self.model(x, seq_length=self.max_seq_length)
            
            # Quitar dimension de batch (max_seq_length, 27)
            generated_seq = generated_seq.squeeze(0).cpu().numpy()
            
        # Reconstruir el formato de diccionario para enviar a Unity
        frames_data = []
        for frame_idx in range(self.max_seq_length):
            frame_features = generated_seq[frame_idx]
            bone_vectors = {}
            
            # Reconstruir los 9 huesos a partir de los 27 valores
            for i, bone in enumerate(expected_bones):
                base_idx = i * 3
                bone_vectors[bone] = {
                    "x": float(frame_features[base_idx]),
                    "y": float(frame_features[base_idx + 1]),
                    "z": float(frame_features[base_idx + 2])
                }
                
            frames_data.append(bone_vectors)
            
        return frames_data
