import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QLineEdit,
                             QGroupBox, QFormLayout, QComboBox, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
import threading

from pose_extraction.detector import PersonDetector
from pose_extraction.landmark_extractor import PoseExtractor
from pose_extraction.mixamo_mapper import MixamoMapper
from ml.dataset_builder import MotionDataset
from ml.trainer import train_model
from ml.predictor import MotionPredictor
import time

SEQUENCES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sequences"))

DARK_BG      = "#0d0d0f"
PANEL_BG     = "#13131a"
CARD_BG      = "#1a1a26"
BORDER       = "#2a2a40"
ACCENT       = "#7c5cbf"
ACCENT_LIGHT = "#9d7de8"
REC_COLOR    = "#e83c3c"
PLAY_COLOR   = "#2ecc71"
STOP_COLOR   = "#e67e22"
PAUSE_COLOR  = "#3498db"
TEXT_PRIMARY = "#f0f0f8"
TEXT_MUTED   = "#7f7f9a"

GLOBAL_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 18px;
    padding: 10px;
    font-size: 13px;
    font-weight: bold;
    color: {ACCENT_LIGHT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    top: -2px;
    padding: 0 6px;
    color: {ACCENT_LIGHT};
}}
QPushButton {{
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: bold;
    font-size: 13px;
    border: none;
    color: {TEXT_PRIMARY};
    background-color: {CARD_BG};
}}
QPushButton:hover {{ opacity: 0.85; }}
QPushButton:disabled {{
    background-color: #1e1e2e;
    color: {TEXT_MUTED};
}}
QLineEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    color: {TEXT_PRIMARY};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    color: {TEXT_PRIMARY};
    min-height: 32px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT};
}}
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
"""


def make_btn(text, color, min_h=44):
    btn = QPushButton(text)
    btn.setMinimumHeight(min_h)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {color};
            color: #ffffff;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            border: none;
            padding: 8px 14px;
        }}
        QPushButton:hover {{
            background-color: {color}cc;
        }}
        QPushButton:disabled {{
            background-color: #2a2a3a;
            color: #55556a;
        }}
    """)
    return btn


class MainWindow(QMainWindow):
    training_finished_signal = pyqtSignal(str)
    camera_scan_signal       = pyqtSignal(str)  # "0,1,2" — índices de cámaras disponibles

    def __init__(self, udp_server=None):
        super().__init__()
        self.udp_server = udp_server
        self.setWindowTitle("DanceLabo — Motion Capture Studio")
        self.setGeometry(100, 60, 1100, 720)
        self.setStyleSheet(GLOBAL_STYLE)

        # Modelos
        self.detector = PersonDetector()
        self.extractor = PoseExtractor()
        self.mapper = MixamoMapper()
        self.dataset = MotionDataset()

        # Variables de video
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Estado
        self.is_recording = False
        self.recorded_frames = []
        self.is_camera = False
        self.yolo_frame_count = 0
        self.last_yolo_box = None
        self.is_playing_back = False
        self.selected_camera_index = 0   # Índice de la cámara activa
        self.live_eval_active = False    # Modo evaluación en tiempo real
        self.live_evaluator = None       # Instancia de LiveEvaluator

        # Timer para tiempo transcurrido
        self._elapsed_seconds = 0
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        # Timer para parpadeo REC
        self._blink_state = False
        self._blink_timer = QTimer()
        self._blink_timer.timeout.connect(self._blink_rec)

        self._init_ui()
        self.training_finished_signal.connect(self.on_training_finished)
        self.camera_scan_signal.connect(self._update_camera_list)
        self._refresh_cameras()          # Detectar cámaras disponibles al arrancar

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Columna izquierda: video ───────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        # Cabecera
        header = QLabel("🎬  DanceLabo Motion Studio")
        header.setStyleSheet(f"font-size:20px; font-weight:bold; color:{ACCENT_LIGHT}; padding:4px 0;")
        left.addWidget(header)

        # Video
        self.video_label = QLabel("Carga un video o activa la cámara para comenzar")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color:{PANEL_BG}; color:{TEXT_MUTED}; font-size:15px;"
            f" border:2px solid {BORDER}; border-radius:10px;"
        )
        self.video_label.setMinimumSize(680, 460)
        left.addWidget(self.video_label, stretch=1)

        # Controles de fuente
        src_row = QHBoxLayout()
        src_row.setSpacing(8)
        self.btn_load   = make_btn("📂  Cargar Video",   ACCENT)
        self.btn_camera = make_btn("📷  Usar Cámara",    "#1a6b8a")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_camera.clicked.connect(self.use_camera)
        src_row.addWidget(self.btn_load)
        src_row.addWidget(self.btn_camera)
        left.addLayout(src_row)

        # ── Selector de cámara ────────────────────────────────────────
        cam_row = QHBoxLayout()
        cam_row.setSpacing(6)
        cam_lbl = QLabel("Cámara:")
        cam_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px; min-width:55px;")
        self.combo_camera = QComboBox()
        self.combo_camera.setFixedHeight(34)
        self.combo_camera.setToolTip(
            "Selecciona la cámara a usar.\n"
            "Si tienes webcam integrada y una externa, aparecerán aquí.\n"
            "Pulsa 🔄 para volver a detectar cámaras."
        )
        self.combo_camera.currentIndexChanged.connect(self._on_camera_selected)
        btn_scan = QPushButton("🔄")
        btn_scan.setFixedSize(34, 34)
        btn_scan.setToolTip("Volver a detectar cámaras disponibles")
        btn_scan.setStyleSheet(f"""QPushButton {{
            background:{CARD_BG}; border:1px solid {BORDER};
            border-radius:6px; color:{ACCENT_LIGHT}; font-size:15px;
        }} QPushButton:hover {{ background:{ACCENT}; color:white; }}""")
        btn_scan.clicked.connect(self._refresh_cameras)
        cam_row.addWidget(cam_lbl)
        cam_row.addWidget(self.combo_camera, stretch=1)
        cam_row.addWidget(btn_scan)
        left.addLayout(cam_row)

        # ── Etiqueta de score en vivo (oculta hasta activar evaluación) ─
        self.lbl_live_score = QLabel("")
        self.lbl_live_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_live_score.setFixedHeight(44)
        self.lbl_live_score.setVisible(False)
        self.lbl_live_score.setStyleSheet(
            "font-size:18px; font-weight:bold; color:#2ecc71;"
            " background:rgba(0,0,0,0.55); border-radius:8px; padding:4px 14px;"
        )
        left.addWidget(self.lbl_live_score)

        # ── Columna derecha: controles ─────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)
        right.setContentsMargins(0, 0, 0, 0)

        # -- Panel de estado --------------------------------------------
        status_group = QGroupBox("Estado")
        sg_lay = QVBoxLayout(status_group)
        sg_lay.setSpacing(8)

        # Indicador de estado (pill)
        self.lbl_status_pill = QLabel("⏹  Detenido")
        self.lbl_status_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_pill.setFixedHeight(38)
        self._set_status_pill("stopped")
        sg_lay.addWidget(self.lbl_status_pill)

        # Tiempo transcurrido
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("⏱  Tiempo:"))
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setStyleSheet(
            f"font-size:22px; font-weight:bold; color:{ACCENT_LIGHT};"
            f" font-family:'Courier New', monospace;"
        )
        time_row.addWidget(self.lbl_time)
        time_row.addStretch()
        sg_lay.addLayout(time_row)

        # Indicador REC parpadeante
        self.lbl_rec_dot = QLabel("● REC")
        self.lbl_rec_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_rec_dot.setStyleSheet(
            f"font-size:13px; font-weight:bold; color:{REC_COLOR};"
            " background:transparent;"
        )
        self.lbl_rec_dot.setVisible(False)
        sg_lay.addWidget(self.lbl_rec_dot)

        right.addWidget(status_group)

        # -- Panel Transport (REC / STOP / PLAY / PAUSE) ----------------
        transport_group = QGroupBox("Control de Transporte")
        tg_lay = QVBoxLayout(transport_group)
        tg_lay.setSpacing(8)

        row1 = QHBoxLayout(); row1.setSpacing(8)
        self.btn_rec   = make_btn("⏺  REC",    REC_COLOR,  50)
        self.btn_stop  = make_btn("⏹  STOP",   STOP_COLOR, 50)
        self.btn_rec.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        row1.addWidget(self.btn_rec)
        row1.addWidget(self.btn_stop)
        tg_lay.addLayout(row1)

        row2 = QHBoxLayout(); row2.setSpacing(8)
        self.btn_play  = make_btn("▶  PLAY",   PLAY_COLOR, 50)
        self.btn_pause = make_btn("⏸  PAUSE",  PAUSE_COLOR, 50)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        row2.addWidget(self.btn_play)
        row2.addWidget(self.btn_pause)
        tg_lay.addLayout(row2)

        right.addWidget(transport_group)

        # -- Selector de secuencias ------------------------------------
        seq_group = QGroupBox("Secuencia")
        seq_lay = QVBoxLayout(seq_group)
        seq_lay.setSpacing(8)

        lbl_seq = QLabel("Nombre del baile:")
        lbl_seq.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px;")
        seq_lay.addWidget(lbl_seq)

        self.txt_dance_name = QLineEdit()
        self.txt_dance_name.setPlaceholderText("Ej: Macarena")
        seq_lay.addWidget(self.txt_dance_name)

        lbl_drop = QLabel("Secuencias disponibles:")
        lbl_drop.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px;")
        seq_lay.addWidget(lbl_drop)

        drop_row = QHBoxLayout(); drop_row.setSpacing(6)
        self.combo_sequences = QComboBox()
        self.combo_sequences.currentTextChanged.connect(self._on_sequence_selected)
        self._refresh_sequences()
        drop_row.addWidget(self.combo_sequences, stretch=1)

        btn_refresh = QPushButton("↺")
        btn_refresh.setFixedSize(34, 34)
        btn_refresh.setToolTip("Recargar lista")
        btn_refresh.setStyleSheet(f"""
            QPushButton {{
                background:{CARD_BG}; border:1px solid {BORDER};
                border-radius:6px; color:{ACCENT_LIGHT}; font-size:16px;
            }}
            QPushButton:hover {{ background:{ACCENT}; color:white; }}
        """)
        btn_refresh.clicked.connect(self._refresh_sequences)
        drop_row.addWidget(btn_refresh)
        seq_lay.addLayout(drop_row)

        right.addWidget(seq_group)

        # -- Panel ML --------------------------------------------------
        ml_group = QGroupBox("Inteligencia Artificial")
        ml_lay = QVBoxLayout(ml_group)
        ml_lay.setSpacing(8)

        self.btn_train     = make_btn("🧠  Entrenar Modelo",    "#1a4a1a")
        self.btn_play_unity = make_btn("🧠  Reproducir (LSTM)", "#004a4a")
        self.btn_play_raw   = make_btn("📼  Secuencia Exacta",  "#003366")

        self.btn_train.setStyleSheet(self.btn_train.styleSheet().replace("#1a4a1a", "#1d6b2e"))
        self.btn_play_unity.setStyleSheet(self.btn_play_unity.styleSheet().replace("#004a4a", "#0e6b6b"))
        self.btn_play_raw.setStyleSheet(self.btn_play_raw.styleSheet().replace("#003366", "#14518a"))

        self.btn_train.clicked.connect(self.start_training)
        self.btn_play_unity.clicked.connect(self.play_in_unity)
        self.btn_play_raw.clicked.connect(self.play_raw_in_unity)
        self.btn_play_raw.setToolTip(
            "Reproduce la grabación original frame a frame — sin pasar por el modelo LSTM."
        )

        ml_lay.addWidget(self.btn_train)
        ml_lay.addWidget(self.btn_play_unity)
        ml_lay.addWidget(self.btn_play_raw)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{BORDER}; margin:4px 0;")
        ml_lay.addWidget(sep)

        self.btn_live_eval = make_btn("🎯  Evaluar en Vivo", "#5a1a8a")
        self.btn_live_eval.setToolTip(
            "Activa evaluación en tiempo real.\n"
            "Compara tu pose (cámara) contra la referencia LSTM del movimiento elegido."
        )
        self.btn_live_eval.clicked.connect(self._toggle_live_eval)
        ml_lay.addWidget(self.btn_live_eval)

        right.addWidget(ml_group)

        # -- Log de estado --------------------------------------------
        log_group = QGroupBox("Log")
        log_lay = QVBoxLayout(log_group)
        self.lbl_record_status = QLabel("Listo.")
        self.lbl_record_status.setWordWrap(True)
        self.lbl_record_status.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:12px; padding:4px;"
        )
        log_lay.addWidget(self.lbl_record_status)
        right.addWidget(log_group)

        right.addStretch()

        # ── Ensamblar ──────────────────────────────────────────────────
        root.addLayout(left, stretch=3)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(320)
        right_widget.setStyleSheet(f"background:{PANEL_BG}; border-radius:10px;")
        root.addWidget(right_widget)

    # ------------------------------------------------------------------
    # Helpers visuales
    # ------------------------------------------------------------------

    def _set_status_pill(self, state: str):
        """state: 'stopped' | 'recording' | 'playing'"""
        styles = {
            "stopped":   (f"background:{CARD_BG}; color:{TEXT_MUTED};",   "⏹  Detenido"),
            "recording": (f"background:{REC_COLOR}22; color:{REC_COLOR}; border:1px solid {REC_COLOR};", "⏺  Grabando"),
            "playing":   (f"background:{PLAY_COLOR}22; color:{PLAY_COLOR}; border:1px solid {PLAY_COLOR};", "▶  Reproduciendo"),
        }
        style, text = styles.get(state, styles["stopped"])
        self.lbl_status_pill.setText(text)
        self.lbl_status_pill.setStyleSheet(
            f"font-size:14px; font-weight:bold; border-radius:8px; padding:4px 12px; {style}"
        )

    def _tick_elapsed(self):
        self._elapsed_seconds += 1
        m, s = divmod(self._elapsed_seconds, 60)
        self.lbl_time.setText(f"{m:02d}:{s:02d}")

    def _blink_rec(self):
        self._blink_state = not self._blink_state
        color = REC_COLOR if self._blink_state else "transparent"
        self.lbl_rec_dot.setStyleSheet(
            f"font-size:13px; font-weight:bold; color:{color}; background:transparent;"
        )

    def _refresh_sequences(self):
        self.combo_sequences.blockSignals(True)
        self.combo_sequences.clear()
        self.combo_sequences.addItem("— Selecciona una secuencia —")
        seq_path = os.path.abspath(SEQUENCES_DIR)
        if os.path.isdir(seq_path):
            files = sorted(f for f in os.listdir(seq_path) if f.endswith(".json"))
            for f in files:
                name = os.path.splitext(f)[0]
                self.combo_sequences.addItem(name)
        self.combo_sequences.blockSignals(False)

    def _on_sequence_selected(self, text):
        if text and not text.startswith("—"):
            # Buscar el archivo .json para leer el movement_name real guardado dentro
            import json
            fpath = os.path.join(SEQUENCES_DIR, f"{text}.json")
            if os.path.isfile(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    movement_name = data.get("movement_name")
                    if movement_name:
                        self.txt_dance_name.setText(movement_name)
                        return
                except Exception as e:
                    print(f"Error leyendo movement_name de {text}.json: {e}")
            
            # Fallback usando regex por si falla la lectura:
            # extrae el nombre base quitando el sufijo _[0-9]+ (ej: Prueba2_1 -> Prueba2)
            import re
            base = re.sub(r'_\d+$', '', text)
            self.txt_dance_name.setText(base)

    # ------------------------------------------------------------------
    # Grabacion
    # ------------------------------------------------------------------

    def start_recording(self):
        name = self.txt_dance_name.text().strip()
        if not name:
            self.lbl_record_status.setText("⚠ Ponle nombre primero!")
            return
        self.mapper.reset_cache()
        self.recorded_frames = []
        self.is_recording = True
        self.btn_rec.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._elapsed_seconds = 0
        self.lbl_time.setText("00:00")
        self._elapsed_timer.start(1000)
        self._blink_timer.start(500)
        self.lbl_rec_dot.setVisible(True)
        self._set_status_pill("recording")
        self.lbl_record_status.setText(f"Grabando '{name}'...")

    def stop_recording(self):
        self.is_recording = False
        self.btn_rec.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._elapsed_timer.stop()
        self._blink_timer.stop()
        self.lbl_rec_dot.setVisible(False)
        self._set_status_pill("stopped")
        self.mapper.reset_cache()
        name = self.txt_dance_name.text().strip()
        if len(self.recorded_frames) > 0:
            self.dataset.save_sequence(name, self.recorded_frames)
            self.lbl_record_status.setText(
                f"✅ Guardado '{name}' — {len(self.recorded_frames)} frames."
            )
            self._refresh_sequences()
        else:
            self.lbl_record_status.setText("Cancelado: No hay frames grabados.")

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def start_training(self):
        self.btn_train.setEnabled(False)
        self.lbl_record_status.setText("🧠 Entrenando... (revisa consola)")
        thread = threading.Thread(target=self._run_training)
        thread.daemon = True
        thread.start()

    def _run_training(self):
        try:
            train_model()
            self.training_finished_signal.emit("✅ Entrenamiento completado. Modelo en data/motion_model.pt")
        except Exception as e:
            import traceback
            print(f"[Error entrenamiento]\n{traceback.format_exc()}")
            self.training_finished_signal.emit(f"❌ Error: {e}")

    def on_training_finished(self, msg: str):
        # Mensaje interno: evaluación en vivo lista
        if msg.startswith("__live_eval_on__:"):
            name = msg.split(":", 1)[1]
            self.btn_live_eval.setText("⏹  Detener Evaluación")
            self.btn_live_eval.setEnabled(True)
            self.lbl_live_score.setVisible(True)
            self.lbl_record_status.setText(
                f"🎯 Evaluando '{name}' en vivo — muévete frente a la cámara."
            )
            # Asegurarse de que la cámara está activa
            if not (self.video_capture and self.video_capture.isOpened()):
                self.use_camera()
            return

        self.lbl_record_status.setText(msg)
        self.btn_train.setEnabled(True)
        self.btn_play_unity.setEnabled(True)
        self.btn_play_raw.setEnabled(True)
        self._set_status_pill("stopped")
        self._elapsed_timer.stop()
        self._blink_timer.stop()
        self.lbl_rec_dot.setVisible(False)

    # ------------------------------------------------------------------
    # Reproducción LSTM
    # ------------------------------------------------------------------

    def play_in_unity(self):
        name = self.txt_dance_name.text().strip()
        if not name:
            self.lbl_record_status.setText("⚠ Ponle nombre al movimiento!")
            return
        self.btn_play_unity.setEnabled(False)
        self.lbl_record_status.setText(f"[LSTM] Reproduciendo '{name}'...")
        self.is_playing_back = True
        self._elapsed_seconds = 0
        self.lbl_time.setText("00:00")
        self._elapsed_timer.start(1000)
        self._set_status_pill("playing")
        thread = threading.Thread(target=self._run_playback, args=(name,))
        thread.daemon = True
        thread.start()

    def _run_playback(self, name):
        try:
            predictor = MotionPredictor()
            frames = predictor.predict_sequence(name)
            if not frames:
                self.training_finished_signal.emit(
                    f"[LSTM] Error: '{name}' no reconocido. ¿Está entrenado?"
                )
                return
            if self.udp_server:
                for frame in frames:
                    self.udp_server.send_pose(frame)
                    time.sleep(1.0 / 30.0)
            self.training_finished_signal.emit(f"[LSTM] '{name}' finalizada.")
        except Exception as e:
            self.training_finished_signal.emit(f"Error LSTM: {e}")
        finally:
            self.is_playing_back = False
            self.btn_play_unity.setEnabled(True)
            self._elapsed_timer.stop()

    # ------------------------------------------------------------------
    # Reproducción directa JSON
    # ------------------------------------------------------------------

    def play_raw_in_unity(self):
        name = self.txt_dance_name.text().strip()
        if not name:
            self.lbl_record_status.setText("⚠ Ponle nombre al movimiento!")
            return
        self.btn_play_raw.setEnabled(False)
        self.lbl_record_status.setText(f"[Exacto] Reproduciendo '{name}'...")
        self.is_playing_back = True
        self._elapsed_seconds = 0
        self.lbl_time.setText("00:00")
        self._elapsed_timer.start(1000)
        self._set_status_pill("playing")
        thread = threading.Thread(target=self._run_raw_playback, args=(name,))
        thread.daemon = True
        thread.start()

    def _run_raw_playback(self, name):
        try:
            predictor = MotionPredictor()
            frames = predictor.playback_raw_sequence(name)
            if not frames:
                self.training_finished_signal.emit(
                    f"[Exacto] No se encontró secuencia para '{name}'."
                )
                return
            if self.udp_server:
                for frame in frames:
                    self.udp_server.send_pose(frame)
                    time.sleep(1.0 / 30.0)
            self.training_finished_signal.emit(
                f"[Exacto] '{name}' finalizada ({len(frames)} frames)."
            )
        except Exception as e:
            self.training_finished_signal.emit(f"Error Exacto: {e}")
        finally:
            self.is_playing_back = False
            self.btn_play_raw.setEnabled(True)
            self._elapsed_timer.stop()

    # ------------------------------------------------------------------
    # Video / Cámara
    # ------------------------------------------------------------------

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Abrir Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if filename:
            if self.video_capture:
                self.video_capture.release()
            self.video_capture = cv2.VideoCapture(filename)
            self.is_camera = False
            self.play_video()

    def use_camera(self):
        if self.video_capture:
            self.video_capture.release()
        idx = self.selected_camera_index
        self.video_capture = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(idx)
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.video_capture.isOpened():
            self.lbl_record_status.setText(f"❌ No se pudo abrir la cámara {idx}.")
            return
        self.lbl_record_status.setText(f"📷 Usando cámara {idx}.")
        self.is_camera = True
        self.play_video()

    def play_video(self):
        if self.video_capture and self.video_capture.isOpened():
            self.timer.start(33)

    def pause_video(self):
        self.timer.stop()

    def update_frame(self):
        if self.is_camera:
            for _ in range(4):
                self.video_capture.grab()
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            if self.is_recording:
                self.stop_recording()
            return
        if self.is_camera:
            frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (800, 600))
        original_shape = frame.shape
        if self.yolo_frame_count % 10 == 0 or self.last_yolo_box is None:
            box = self.detector.detect_person(frame)
            if box is not None:
                self.last_yolo_box = box
        else:
            box = self.last_yolo_box
        self.yolo_frame_count += 1
        cropped, offset = self.detector.crop_person(frame, box)
        if box is not None:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 100), 2)
            frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            landmarks, results = self.extractor.extract_pose(frame_rgb, offset, original_shape)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    px_x = int(lm.x * cropped.shape[1]) + offset[0]
                    px_y = int(lm.y * cropped.shape[0]) + offset[1]
                    cv2.circle(frame, (px_x, px_y), 4, (0, 200, 255), -1)
                bone_vectors = self.mapper.get_bone_vectors(landmarks)
                if self.udp_server and not self.is_playing_back:
                    self.udp_server.send_pose(bone_vectors)
                if self.is_recording:
                    self.recorded_frames.append(bone_vectors)
                    self.lbl_record_status.setText(
                        f"Grabando... ({len(self.recorded_frames)} frames)"
                    )
                # Evaluación en vivo — compara frame a frame contra referencia LSTM
                if self.live_eval_active and self.live_evaluator and bone_vectors:
                    self._run_live_eval_frame(bone_vectors, frame)
        frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb_display.shape
        q_img = QImage(frame_rgb_display.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    # ------------------------------------------------------------------
    # Selector de cámara — detección automática
    # ------------------------------------------------------------------

    def _refresh_cameras(self):
        """Detecta cámaras disponibles en un hilo separado para no bloquear la UI."""
        self.lbl_record_status.setText("🔍 Detectando cámaras...")
        t = threading.Thread(target=self._do_scan_cameras, daemon=True)
        t.start()

    def _do_scan_cameras(self):
        """Hilo: prueba índices 0-5 con DirectShow; fallback sin backend si falla."""
        available = []
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        if not available:
            # Fallback sin DirectShow (Linux / Mac / cámaras virtuales)
            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
        result = ",".join(str(c) for c in available) if available else "none"
        self.camera_scan_signal.emit(result)

    def _update_camera_list(self, cameras_str: str):
        """Actualiza el combo de cámaras (siempre en el hilo principal vía signal)."""
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()
        if cameras_str == "none":
            self.combo_camera.addItem("❌ Sin cámaras detectadas", userData=None)
            self.lbl_record_status.setText("❌ No se detectó ninguna cámara.")
        else:
            indices = [int(c) for c in cameras_str.split(",") if c.isdigit()]
            for idx in indices:
                self.combo_camera.addItem(f"📷 Cámara {idx}", userData=idx)
            # Restaurar la selección anterior si todavía existe
            for i in range(self.combo_camera.count()):
                if self.combo_camera.itemData(i) == self.selected_camera_index:
                    self.combo_camera.setCurrentIndex(i)
                    break
            n = len(indices)
            self.lbl_record_status.setText(
                f"✅ {n} cámara{'s' if n != 1 else ''} detectada{'s' if n != 1 else ''}."
            )
        self.combo_camera.blockSignals(False)

    def _on_camera_selected(self, index: int):
        """Cambia la cámara activa cuando el usuario elige una diferente en el combo."""
        data = self.combo_camera.itemData(index)
        if data is None:
            return
        prev = self.selected_camera_index
        self.selected_camera_index = data
        # Si la cámara ya está corriendo, cambiarla en vivo
        if self.is_camera and self.timer.isActive() and prev != data:
            self.use_camera()

    # ------------------------------------------------------------------
    # Evaluación en vivo (E-04)
    # ------------------------------------------------------------------

    def _toggle_live_eval(self):
        """Alterna entre activar y detener la evaluación en tiempo real."""
        if self.live_eval_active:
            self._stop_live_eval()
        else:
            self._start_live_eval()

    def _start_live_eval(self):
        name = self.txt_dance_name.text().strip()
        if not name:
            self.lbl_record_status.setText("⚠ Escribe el nombre del movimiento a evaluar.")
            return
        self.btn_live_eval.setEnabled(False)
        self.lbl_record_status.setText(f"🔄 Cargando referencia '{name}'...")
        t = threading.Thread(target=self._do_load_live_eval, args=(name,), daemon=True)
        t.start()

    def _do_load_live_eval(self, name: str):
        """Hilo: carga el modelo y genera la secuencia de referencia."""
        try:
            from ml.live_evaluator import LiveEvaluator
            evaluator = LiveEvaluator(name)
            if not evaluator.is_ready:
                self.training_finished_signal.emit(
                    f"❌ '{name}' no encontrado. ¿Está el modelo entrenado con este movimiento?"
                )
                return
            self.live_evaluator = evaluator
            self.live_eval_active = True
            self.training_finished_signal.emit(f"__live_eval_on__:{name}")
        except Exception as e:
            self.training_finished_signal.emit(f"❌ Error iniciando evaluación: {e}")

    def _stop_live_eval(self):
        self.live_eval_active = False
        self.live_evaluator   = None
        self.lbl_live_score.setVisible(False)
        self.btn_live_eval.setText("🎯  Evaluar en Vivo")
        self.btn_live_eval.setEnabled(True)
        self.btn_live_eval.setStyleSheet(
            self.btn_live_eval.styleSheet()
        )
        self.lbl_record_status.setText("Evaluación en vivo detenida.")

    def _run_live_eval_frame(self, bone_vectors: dict, frame):
        """Evalúa un frame y dibuja el score sobre la imagen de OpenCV."""
        result = self.live_evaluator.evaluate(bone_vectors)
        if not result:
            return

        overall    = result.get("overall", 0.0)
        color_name = result.get("overall_color", "rojo")
        pct        = int(overall * 100)

        # Colores BGR para cv2
        cv_colors = {"verde": (0, 220, 80), "amarillo": (0, 200, 255), "rojo": (50, 50, 230)}
        qt_colors = {"verde": "#2ecc71",    "amarillo": "#f39c12",     "rojo": "#e74c3c"}
        cv_color  = cv_colors.get(color_name, (255, 255, 255))
        qt_color  = qt_colors.get(color_name, "#ffffff")

        # Fondo semitransparente + score principal
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (270, 48), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Score: {pct}%  [{color_name.upper()}]",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, cv_color, 2)

        # Score por segmento (mini labels)
        seg_labels = {
            "brazo_izquierdo":  "Brazo Izq",
            "brazo_derecho":    "Brazo Der",
            "pierna_izquierda": "Pierna Izq",
            "pierna_derecha":   "Pierna Der",
            "torso":            "Torso",
        }
        seg_cv_colors = {"verde": (0,220,80), "amarillo": (0,200,255), "rojo": (50,50,230)}
        segments = result.get("segments", {})
        colors   = result.get("colors", {})
        y = 64
        for seg, val in segments.items():
            if val is None:
                continue
            sc = seg_cv_colors.get(colors.get(seg, "rojo"), (180, 180, 180))
            cv2.putText(frame,
                        f"{seg_labels.get(seg, seg)}: {int(val * 100)}%",
                        (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sc, 1)
            y += 18

        # Barra de progreso de la secuencia
        progress = self.live_evaluator.get_progress_pct()
        bar_w = int(250 * progress)
        cv2.rectangle(frame, (14, y + 4), (264, y + 12), (50, 50, 60), -1)
        cv2.rectangle(frame, (14, y + 4), (14 + bar_w, y + 12), cv_color, -1)

        # Actualizar QLabel de score (ya estamos en el hilo del QTimer → hilo principal)
        self.lbl_live_score.setText(f"🎯  Score en vivo: {pct}%")
        self.lbl_live_score.setStyleSheet(
            f"font-size:18px; font-weight:bold; color:{qt_color};"
            f" background:rgba(0,0,0,0.55); border-radius:8px; padding:4px 14px;"
        )

        # Enviar score a Unity por UDP (para MovementComparator.cs → FeedbackUI.cs)
        # Los campos __score_*__ son ignorados por MixamoAnimator y leídos por MovementComparator
        if self.udp_server:
            score_packet = {
                "__score_overall__":       round(overall, 4),
                "__score_brazo_izquierdo__": round(segments.get("brazo_izquierdo")  or 0.0, 4),
                "__score_brazo_derecho__":   round(segments.get("brazo_derecho")    or 0.0, 4),
                "__score_pierna_izquierda__":round(segments.get("pierna_izquierda") or 0.0, 4),
                "__score_pierna_derecha__":  round(segments.get("pierna_derecha")   or 0.0, 4),
                "__score_torso__":           round(segments.get("torso")            or 0.0, 4),
            }
            self.udp_server.send_pose(score_packet)

    # ------------------------------------------------------------------
    # Cierre de ventana
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._stop_live_eval()
        if self.video_capture:
            self.video_capture.release()
        if self.udp_server:
            self.udp_server.stop()
        event.accept()
