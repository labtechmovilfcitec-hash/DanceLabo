# Dance Labo — Motion Capture Studio

Sistema de captura y evaluación de movimiento de baile usando YOLO + MediaPipe + LSTM + Unity.

## Descripción

Dance Labo permite:
- **Grabar** movimientos de un robot 3D en Unity como secuencias de referencia.
- **Entrenar** un modelo LSTM que aprende esos movimientos.
- **Evaluar en tiempo real** si un estudiante frente a la cámara replica correctamente los movimientos.
- **Visualizar** el feedback con colores verde/amarillo/rojo por segmento corporal (Unity + UI Python).

## Requisitos del sistema

- Python 3.9 – 3.12
- Unity 2022.x con paquete Newtonsoft.Json
- Webcam o cámara USB
- (Recomendado) CPU Intel i5 o superior para 30 FPS en tiempo real

## Instalación

```bash
cd motion_ml_app
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Verificar el entorno (A-01)

```bash
python check_env.py
```

Verifica que todas las dependencias estén instaladas y que los archivos del modelo existen.

## Uso

### 1. Iniciar la aplicación

```bash
cd motion_ml_app
python main.py
```

### 2. Grabar una secuencia de referencia

1. En Unity: pon el avatar en **Play Mode** con `MixamoAnimator.cs` activo.
2. En la UI de Python:
   - Escribe el nombre del baile (ej. `Secuencia1`).
   - Pulsa **REC** (el botón principal cambiará a **STOP** y se habilitará el botón **CANCELAR** al lado).
   - Realiza los movimientos frente a la cámara (un punto rojo `REC` parpadeará al lado del título principal).
   - Pulsa **STOP** para guardar el JSON en `data/sequences/` (o pulsa **CANCELAR** para abortar y descartar los datos capturados).
3. **Ver Historial de Logs:** Pulsa el icono de libreta de registros en el panel de Estado para abrir la ventana modal y ver de manera detallada las tareas del backend (entrenamiento PyTorch, carga de secuencias, comunicación UDP) sin spam de frames.
4. **Eliminar Secuencias:** Selecciona la secuencia en el desplegable y haz clic en el icono del bote de basura rojo. Confirma el borrado haciendo clic en **Sí** en la alerta y escribiendo el nombre exacto de la secuencia como doble paso de seguridad.

### 3. Entrenar el modelo

```bash
python train.py
```

El modelo se guarda en data/motion_model.pt.

### 4. Evaluar en tiempo real

1. En Unity: **Play Mode** con MovementComparator.cs y FeedbackUI.cs configurados.
2. En Python UI:
   - Selecciona la cámara correcta.
   - Escribe el nombre del movimiento (ej. Secuencia1).
   - Pulsa **Usar Cámara**.
   - Pulsa **Evaluar en Vivo**.

### 5. Validar el modelo

```bash
python inference.py     # valida todos los movimientos contra datos reales
python benchmark.py     # mide latencia de inferencia (debe ser < 100ms)
```

## Estructura del proyecto

```text
motion_ml_app/
├── main.py               # Punto de entrada
├── train.py              # Entrenar modelo LSTM
├── inference.py          # Validar modelo
├── benchmark.py          # Medir rendimiento (E-05)
├── check_env.py          # Verificar entorno (A-01)
├── requirements.txt      # Dependencias Python
├── ui/                   # Interfaz gráfica (PyQt6)
├── ml/                   # Modelo LSTM, scoring, evaluación
├── pose_extraction/      # YOLO + MediaPipe + MixamoMapper
├── communication/        # UDP server + scripts C# de Unity
└── data/                 # Modelo entrenado y secuencias

Assets/ (Unity)
├── MixamoAnimator.cs     # Mueve el robot con los datos de Python
├── MovementComparator.cs # Recibe scores de Python y dispara eventos
├── FeedbackUI.cs         # Muestra colores verde/amarillo/rojo en la UI
├── UDPClient.cs          # Comunicación UDP Python ↔ Unity
└── LABO new rig.fbx      # Nuevo avatar con rig completo (P3)
```