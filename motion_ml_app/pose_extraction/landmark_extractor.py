import cv2
import mediapipe as mp
import numpy as np

class PoseExtractor:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1, # 1 es el modelo BlazePose completo con precisión métrica 3D
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        # Detector de corto alcance (model_selection=0, ideal para planos medios y recortes de persona)
        self.face_detector_short = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.35,
            model_selection=0
        )
        # Detector de rango completo (model_selection=1, ideal cuando la persona está a más de 2 metros)
        self.face_detector_full = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.35,
            model_selection=1
        )
        
        # Historial de orientación con histéresis responsiva (ventana de 10 frames ~ 0.3s)
        # Responde al instante en giros 360° sin parpadeos ni falsos positivos en vistas de perfil
        self._face_detected_history = []
        self._is_facing_backward = False
        self._history_len = 10

    def extract_pose(self, frame_rgb, offset=(0, 0), original_shape=None, full_frame_rgb=None):
        """
        Extrae la pose del frame (generalmente el area recortada por YOLO).
        Detecta también si la persona está de espaldas para orientar al personaje en 360°.
        """
        results = self.pose.process(frame_rgb)
        landmarks_data = {}

        # Detección de rostro exclusivamente sobre el recorte de la persona (frame_rgb):
        # Primero busca con corto alcance (model_selection=0); si no encuentra, prueba rango completo.
        face_results = self.face_detector_short.process(frame_rgb)
        face_found = False
        if face_results and face_results.detections:
            for det in face_results.detections:
                score = det.score[0] if (det.score and len(det.score) > 0) else 0.0
                if score >= 0.35:
                    box = det.location_data.relative_bounding_box
                    if box.ymin < 0.65:
                        face_found = True
                        break

        if not face_found:
            face_results_full = self.face_detector_full.process(frame_rgb)
            if face_results_full and face_results_full.detections:
                for det in face_results_full.detections:
                    score = det.score[0] if (det.score and len(det.score) > 0) else 0.0
                    if score >= 0.35:
                        box = det.location_data.relative_bounding_box
                        if box.ymin < 0.65:
                            face_found = True
                            break

        is_front_frame = face_found

        self._face_detected_history.append(is_front_frame)
        if len(self._face_detected_history) > self._history_len:
            self._face_detected_history.pop(0)

        # Histéresis responsiva (ventana de 10 frames ~ 0.3s):
        # Para pasar a espaldas: requiere al menos 7 frames continuos sin rostro (front_votes <= 2)
        # Para volver al frente: requiere 5 frames con rostro detectado
        front_votes = sum(1 for f in self._face_detected_history if f)
        if self._is_facing_backward:
            if front_votes >= 5:
                self._is_facing_backward = False
        else:
            if front_votes <= 2 and len(self._face_detected_history) >= 8:
                self._is_facing_backward = True

        if results.pose_landmarks and original_shape:
            crop_h, crop_w = frame_rgb.shape[:2]
            full_w = float(original_shape[1])
            off_x, off_y = offset

            for id, lm in enumerate(results.pose_landmarks.landmark):
                px_x = lm.x * crop_w + off_x
                px_y = lm.y * crop_h + off_y
                landmarks_data[id] = {
                    'x': px_x / full_w,
                    'y': px_y / full_w,
                    'z': (lm.z * crop_w) / full_w,
                    'v': lm.visibility
                }

            if results.pose_world_landmarks:
                landmarks_data['world_landmarks'] = {
                    id: {
                        'x': float(wlm.x),
                        'y': float(wlm.y),
                        'z': float(wlm.z),
                        'v': float(wlm.visibility)
                    }
                    for id, wlm in enumerate(results.pose_world_landmarks.landmark)
                }

            # La orientación espacial 360° es manejada en 3D por MixamoMapper y MixamoAnimator.
            # NO se invierten ni intercambian huesos aquí, evitando que las extremidades se crucen o distorsionen.
            landmarks_data['is_backward'] = self._is_facing_backward

        return landmarks_data, results

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
