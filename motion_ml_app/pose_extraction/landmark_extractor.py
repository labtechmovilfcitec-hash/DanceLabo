import cv2
import mediapipe as mp
import numpy as np

class PoseExtractor:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=0, # 0, 1, o 2 (1 es buen balance velocidad/precision)
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_pose(self, frame_rgb, offset=(0, 0), original_shape=None):
        """
        Extrae la pose del frame (generalmente el area recortada por YOLO).
        """
        results = self.pose.process(frame_rgb)
        landmarks_data = {}
        
        if results.pose_landmarks and original_shape:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                # Guardamos x, y, z relativas y visibilidad
                landmarks_data[id] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'v': lm.visibility
                }
                
        return landmarks_data, results

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
