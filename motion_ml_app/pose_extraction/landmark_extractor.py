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
        # Face detector ultraligero para determinar si la persona está de frente o de espaldas
        # model_selection=0 es optimizado para caras dentro de 2 metros (webcam).
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.70,
            model_selection=0
        )
        
        # Historial de orientación con histéresis robusta (ventana de 15 frames ~ 0.5s)
        # Responde con estabilidad en giros 360° sin parpadeos ni falsos positivos en vistas de perfil
        self._face_detected_history = []
        self._is_facing_backward = False
        self._history_len = 15

    def extract_pose(self, frame_rgb, offset=(0, 0), original_shape=None, full_frame_rgb=None):
        """
        Extrae la pose del frame (generalmente el area recortada por YOLO).
        Detecta también si la persona está de espaldas para orientar al personaje en 360°.
        """
        results = self.pose.process(frame_rgb)
        landmarks_data = {}

        # Detección de rostro sobre el recorte de la persona (frame_rgb):
        face_results = self.face_detector.process(frame_rgb)
        face_found = False
        if face_results.detections:
            for det in face_results.detections:
                if det.score and det.score[0] >= 0.70:
                    face_found = True
                    break

        # Validación con landmarks faciales de MediaPipe Pose:
        pose_indicates_front = False
        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            nose = lms[0]
            eye_l = lms[2]
            eye_r = lms[5]

            # La nariz y al menos uno de los ojos deben tener alta confianza visible
            if nose.visibility > 0.70 and (eye_l.visibility > 0.65 or eye_r.visibility > 0.65):
                pose_indicates_front = True

        # Un frame se considera 'de frente' si los landmarks faciales de la pose apuntan a la cámara.
        # Esto impide que reflejos en espejos de fondo o falsos positivos de cabello
        # hagan que el personaje se voltee de frente cuando el usuario sigue de espaldas.
        if results.pose_landmarks:
            is_front_frame = pose_indicates_front
        else:
            is_front_frame = face_found

        self._face_detected_history.append(is_front_frame)
        if len(self._face_detected_history) > self._history_len:
            self._face_detected_history.pop(0)

        # Histéresis robusta (ventana de 15 frames ~ 0.5s):
        # Para pasar a espaldas: requiere 11+ frames sin detección facial (front_votes <= 3 con al menos 12 frames)
        # Para volver al frente: requiere 8+ frames con rostro claramente detectado
        front_votes = sum(1 for f in self._face_detected_history if f)
        if self._is_facing_backward:
            if front_votes >= 8:
                self._is_facing_backward = False
        else:
            if front_votes <= 3 and len(self._face_detected_history) >= 12:
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
