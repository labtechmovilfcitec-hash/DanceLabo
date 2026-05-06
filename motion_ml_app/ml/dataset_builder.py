import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, data_dir="data/sequences"):
        self.data_dir = data_dir
        self.sequences = []
        self.labels = []
        self.label_map = {}
        self.max_length = 0
        
        # Crear directorio si no existe
        os.makedirs(self.data_dir, exist_ok=True)
        self.load_data()

    def load_data(self):
        self.sequences = []
        self.labels = []
        if not os.path.exists(self.data_dir):
            return
            
        label_idx = 0
        for file in os.listdir(self.data_dir):
            if file.endswith('.json'):
                path = os.path.join(self.data_dir, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        
                    movement_name = data.get("movement_name", "unknown")
                    if movement_name not in self.label_map:
                        self.label_map[movement_name] = label_idx
                        label_idx += 1
                    
                    # frame_data: lista de diccionarios con landmarks
                    # Lo aplanamos a [num_frames, 33*3] (x,y,z para 33 landmarks de MediaPipe)
                    frames = data.get("frames", [])
                    seq_array = []
                    for frame in frames:
                        landmarks = frame.get("landmarks", {})
                        frame_features = []
                        # Extraer solo x,y,z en orden (99 valores por frame)
                        for i in range(33):
                            lm = landmarks.get(str(i), {'x': 0.0, 'y': 0.0, 'z': 0.0})
                            frame_features.extend([lm['x'], lm['y'], lm['z']])
                        seq_array.append(frame_features)
                    
                    if len(seq_array) > 0:
                        seq_tensor = torch.tensor(seq_array, dtype=torch.float32)
                        self.sequences.append(seq_tensor)
                        self.labels.append(self.label_map[movement_name])
                        
                        if len(seq_array) > self.max_length:
                            self.max_length = len(seq_array)
                except Exception as e:
                    print(f"Error cargando {file}: {e}")

    def save_sequence(self, movement_name, frames_data):
        """
        frames_data: lista de dicts (el formato devuelto por PoseExtractor)
        """
        filename = f"{movement_name}_{len(self.sequences)}.json"
        path = os.path.join(self.data_dir, filename)
        
        export_data = {
            "movement_name": movement_name,
            "frames": []
        }
        
        for i, frame_landmarks in enumerate(frames_data):
            export_data["frames"].append({
                "frame": i,
                "landmarks": frame_landmarks
            })
            
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=4)
            
        print(f"Movimiento '{movement_name}' guardado exitosamente.")
        # Recargar dataset para incluir la nueva secuencia
        self.load_data()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Aplicamos padding de ceros al final para que todas las secuencias 
        # del batch tengan el mismo tamaño (max_length)
        pad_size = self.max_length - seq.size(0)
        if pad_size > 0:
            padding = torch.zeros(pad_size, seq.size(1))
            seq = torch.cat([seq, padding], dim=0)
            
        label = self.labels[idx]
        return seq, label
