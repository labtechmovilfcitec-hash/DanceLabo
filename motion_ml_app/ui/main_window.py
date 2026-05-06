import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from pose_extraction.detector import PersonDetector
from pose_extraction.landmark_extractor import PoseExtractor
from pose_extraction.mixamo_mapper import MixamoMapper

class MainWindow(QMainWindow):
    def __init__(self, udp_server=None):
        super().__init__()
        self.udp_server = udp_server
        self.setWindowTitle("Motion ML - Captura y Entrenamiento")
        self.setGeometry(100, 100, 1024, 768)
        
        # Modelos
        self.detector = PersonDetector()
        self.extractor = PoseExtractor()
        self.mapper = MixamoMapper()
        
        # Variables de video
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self._init_ui()
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # Area de video
        self.video_label = QLabel("Carga un video para comenzar")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: white; font-size: 16px;")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, stretch=1)
        
        # Controles
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Cargar Video")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_load.setMinimumHeight(40)
        
        self.btn_play = QPushButton("Reproducir")
        self.btn_play.clicked.connect(self.play_video)
        self.btn_play.setMinimumHeight(40)
        
        self.btn_pause = QPushButton("Pausar")
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_pause.setMinimumHeight(40)
        
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_pause)
        
        main_layout.addLayout(control_layout)
        central_widget.setLayout(main_layout)
        
    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if filename:
            if self.video_capture:
                self.video_capture.release()
            self.video_capture = cv2.VideoCapture(filename)
            self.play_video()
            
    def play_video(self):
        if self.video_capture and self.video_capture.isOpened():
            self.timer.start(33) # ~30 fps
            
    def pause_video(self):
        self.timer.stop()
        
    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            return
            
        # Redimensionar para procesamiento mas rapido
        frame = cv2.resize(frame, (800, 600))
        original_shape = frame.shape
        
        # 1. Deteccion con YOLO
        box = self.detector.detect_person(frame)
        cropped, offset = self.detector.crop_person(frame, box)
        
        if box is not None:
            # Dibujar rectangulo verde claro
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 100), 2)
            
            # 2. Extraccion de Pose en el recorte
            frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            landmarks, results = self.extractor.extract_pose(frame_rgb, offset, original_shape)
            
            # Dibujar los landmarks sobre el frame original
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    # Ajustar coord normalizada de mediapipe -> pixeles recorte -> pixeles frame
                    px_x = int(lm.x * cropped.shape[1]) + offset[0]
                    px_y = int(lm.y * cropped.shape[0]) + offset[1]
                    cv2.circle(frame, (px_x, px_y), 4, (0, 200, 255), -1)

                # Enviar a Unity via UDP
                if self.udp_server:
                    bone_vectors = self.mapper.get_bone_vectors(landmarks)
                    self.udp_server.send_pose(bone_vectors)

        # 3. Mostrar frame en UI
        frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb_display.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        if self.video_capture:
            self.video_capture.release()
        if self.udp_server:
            self.udp_server.stop()
        event.accept()
