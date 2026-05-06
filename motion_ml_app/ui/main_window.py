import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from pose_extraction.detector import PersonDetector
from pose_extraction.landmark_extractor import PoseExtractor
from pose_extraction.mixamo_mapper import MixamoMapper
from ml.dataset_builder import MotionDataset

class MainWindow(QMainWindow):
    def __init__(self, udp_server=None):
        super().__init__()
        self.udp_server = udp_server
        self.setWindowTitle("Motion ML - Captura y Entrenamiento")
        self.setGeometry(100, 100, 1200, 768)
        
        # Modelos
        self.detector = PersonDetector()
        self.extractor = PoseExtractor()
        self.mapper = MixamoMapper()
        self.dataset = MotionDataset()
        
        # Variables de video
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Estado de grabacion
        self.is_recording = False
        self.recorded_frames = []
        
        self._init_ui()
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Columna Izquierda: Video y controles de reproduccion
        video_layout = QVBoxLayout()
        
        # Area de video
        self.video_label = QLabel("Carga un video para comenzar")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: white; font-size: 16px;")
        self.video_label.setMinimumSize(800, 600)
        video_layout.addWidget(self.video_label, stretch=1)
        
        # Controles
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("📂 Cargar Video")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_load.setMinimumHeight(40)
        
        self.btn_camera = QPushButton("📷 Usar Cámara")
        self.btn_camera.clicked.connect(self.use_camera)
        self.btn_camera.setMinimumHeight(40)
        
        self.btn_play = QPushButton("▶️ Reproducir")
        self.btn_play.clicked.connect(self.play_video)
        self.btn_play.setMinimumHeight(40)
        
        self.btn_pause = QPushButton("⏸️ Pausar")
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_pause.setMinimumHeight(40)
        
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_camera)
        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_pause)
        
        video_layout.addLayout(control_layout)
        
        # Columna Derecha: Controles de ML y Grabacion
        side_layout = QVBoxLayout()
        
        group_record = QGroupBox("Grabación de Dataset")
        form_layout = QFormLayout()
        
        self.txt_dance_name = QLineEdit()
        self.txt_dance_name.setPlaceholderText("Ej: Macarena")
        form_layout.addRow("Nombre del Baile:", self.txt_dance_name)
        
        self.btn_record = QPushButton("🔴 Grabar Secuencia")
        self.btn_record.clicked.connect(self.start_recording)
        self.btn_record.setStyleSheet("background-color: #550000; color: white;")
        form_layout.addRow(self.btn_record)
        
        self.btn_stop_record = QPushButton("⏹️ Detener / Guardar")
        self.btn_stop_record.clicked.connect(self.stop_recording)
        self.btn_stop_record.setEnabled(False)
        form_layout.addRow(self.btn_stop_record)
        
        self.lbl_record_status = QLabel("Estado: Inactivo")
        form_layout.addRow(self.lbl_record_status)
        
        group_record.setLayout(form_layout)
        side_layout.addWidget(group_record)
        side_layout.addStretch()
        
        # Unir ambas columnas
        main_layout.addLayout(video_layout, stretch=3)
        main_layout.addLayout(side_layout, stretch=1)
        
        central_widget.setLayout(main_layout)
        
    def start_recording(self):
        name = self.txt_dance_name.text().strip()
        if not name:
            self.lbl_record_status.setText("Error: Ponle nombre primero!")
            return
            
        self.recorded_frames = []
        self.is_recording = True
        self.btn_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)
        self.lbl_record_status.setText(f"Grabando... (0 frames)")
        
    def stop_recording(self):
        self.is_recording = False
        self.btn_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        
        name = self.txt_dance_name.text().strip()
        if len(self.recorded_frames) > 0:
            self.dataset.save_sequence(name, self.recorded_frames)
            self.lbl_record_status.setText(f"Guardado '{name}' con {len(self.recorded_frames)} frames.")
        else:
            self.lbl_record_status.setText("Cancelado: No hay frames grabados.")

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if filename:
            if self.video_capture:
                self.video_capture.release()
            self.video_capture = cv2.VideoCapture(filename)
            self.play_video()
            
    def use_camera(self):
        if self.video_capture:
            self.video_capture.release()
        
        # El 0 suele ser la webcam por defecto de la laptop o PC
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.lbl_record_status.setText("Error: No se detectó cámara web.")
            return
            
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
            # Si se acaba el video y estaba grabando, auto detener
            if self.is_recording:
                self.stop_recording()
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

                # Calcular vectores y enviar
                bone_vectors = self.mapper.get_bone_vectors(landmarks)
                if self.udp_server:
                    self.udp_server.send_pose(bone_vectors)
                    
                # Si estamos grabando, guardar en la memoria temporal
                if self.is_recording:
                    self.recorded_frames.append(bone_vectors)
                    self.lbl_record_status.setText(f"Grabando... ({len(self.recorded_frames)} frames)")

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
