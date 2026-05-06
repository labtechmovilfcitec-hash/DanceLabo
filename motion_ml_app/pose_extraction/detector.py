import cv2
import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt'):
        # Carga el modelo YOLOv8 nano preentrenado (es el mas rapido)
        self.model = YOLO(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print("YOLOv8 usando GPU (CUDA)!")
        else:
            print("YOLOv8 usando CPU.")
    
    def detect_person(self, frame):
        """
        Detecta a la persona en el frame y devuelve el bounding box mas grande o centrado.
        """
        results = self.model(frame, classes=[0], verbose=False, device=self.device) # class 0 es 'person'
        
        best_box = None
        max_area = 0
        
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # box.xyxy format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # Seleccionamos a la persona mas grande
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)
                    
        return best_box

    def crop_person(self, frame, box, padding=30):
        """
        Recorta el frame basado en el bounding box con un poco de padding.
        """
        if box is None:
            return frame, (0, 0)
            
        x1, y1, x2, y2 = box
        
        # Aplicar padding sin salirnos de los limites del frame
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        cropped = frame[y1:y2, x1:x2]
        return cropped, (x1, y1) # Retorna el recorte y el offset
