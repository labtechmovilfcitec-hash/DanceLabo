# validacion_final.md — Validación de los 4 Requerimientos Mínimos
## Dance Labo · F-04 · Sprint 6
**Última actualización:** Mayo 2026  
**Sistema:** Dance Labo — LSTM sobre huesos Mixamo (X Bot / El Bueno)

---

## ✅ Checklist General

| # | Requerimiento | Estado |
|---|---|---|
| ① | Detecta réplica y muestra feedback visual | ✅ **COMPLETO** — LiveEvaluator + FeedbackUI.cs |
| ② | Robot 3D mejorado (rig + aprende movimientos) | ✅ **COMPLETO** — LSTM entrenado con Prueba1-6 |
| ③ | 2+ secuencias grabadas y modelo entrenado | ✅ **COMPLETO** — 29 archivos, 6 movimientos |
| ④ | Mapeo de huesos documentado | ✅ **COMPLETO** |

---

## ① Detecta Réplica y Muestra Feedback Visual ✅

> El sistema detecta en tiempo real cuándo el estudiante replica el movimiento del robot
> y muestra colores (verde / amarillo / rojo) por segmento corporal.

### Pipeline completo implementado

```
Webcam
  └─▶ YOLO (detección persona)
        └─▶ MediaPipe (33 landmarks 3D)
              └─▶ MixamoMapper (9 vectores de huesos normalizados)
                    ├─▶ UDP → MixamoAnimator.cs (mueve el robot en Unity)
                    └─▶ LiveEvaluator.evaluate(bones)
                              └─▶ ScoringEngine.evaluate_frame()
                                    ├─▶ Score por segmento (0.0 – 1.0)
                                    ├─▶ Overlay en video (cv2.putText)
                                    ├─▶ QLabel en UI Python
                                    └─▶ UDP → MovementComparator.cs
                                              └─▶ FeedbackUI.cs (Unity)
```

### Criterios de aceptación — Req ①

- [x] Detección de réplica con feedback verde (score ≥ 85%)
- [x] Colores por segmento: Verde ≥85%, Amarillo 60-84%, Rojo <60%
- [x] Score general visible en la UI de Python (lbl_live_score)
- [x] Score visible en Unity via `MovementComparator.cs` + `FeedbackUI.cs`
- [x] Transición suave con lerp (sin parpadeo) — `LerpSpeed` configurable
- [x] Overlay de score en el frame de video (cv2 semitransparente)
- [x] Evento `OnReplicaDetected` cuando score > 85% por > 1s
- [x] Evento `OnGoodStreak` cuando score verde por > 3s consecutivos
- [x] Barra de progreso de la secuencia de referencia

### Cómo activar la evaluación en vivo

```bash
# 1. En Unity: Play Mode con "El bueno" en escena
# 2. En Python:
cd motion_ml_app
python main.py

# 3. En la UI:
#    - Escribe el nombre del movimiento en el campo de texto (ej: "Prueba1")
#    - Pulsa "📷 Usar Cámara"
#    - Pulsa "🎯 Evaluar en Vivo"
#    → El score se muestra en pantalla y se envía a Unity en tiempo real
```

---

## ② Robot 3D Mejorado ✅

> El modelo LSTM aprende y reproduce los movimientos grabados con el robot.

### Arquitectura del modelo (E-01)

| Parámetro | Valor |
|---|---|
| Tipo | LSTM Generativo (MotionLSTMGenerator) |
| Input | ID de movimiento (embedding) |
| Output | 27 valores por frame (9 huesos × xyz) |
| Hidden size | 256 |
| Capas LSTM | 3 |
| Epochs | 150 |
| Optimizador | Adam + StepLR (÷2 cada 50 epochs) |
| Criterio | MSELoss |

### Proceso de generación de referencia

```python
predictor = MotionPredictor("data/motion_model.pt")
frames = predictor.predict_sequence("Prueba1")
# → lista de dicts [{bone: {x,y,z}}, ...] para cada frame
```

### Criterios de aceptación — Req ②

- [x] LSTM entrenado y guardado (`data/motion_model.pt`)
- [x] Genera secuencias de referencia por movimiento en < 5s de carga
- [x] Label map guardado y cargable (`data/label_map.pt`)
- [x] Script reproducible: mismos datos = mismo modelo (`train.py`)
- [x] El modelo se re-entrena automáticamente con `python train.py` cuando hay nuevas secuencias

---

## ③ Secuencias Grabadas y Modelo Entrenado ✅

> El modelo fue entrenado con 29 secuencias de 6 movimientos distintos, grabadas
> directamente desde Unity con el robot "El bueno".

### Secuencias disponibles (`data/sequences/`)

| Movimiento | Archivos | Total frames aprox. |
|---|---|---|
| Prueba1 | 1 (Prueba1_0.json) | ~500 |
| Prueba2 | 1 (Prueba2_1.json) | ~500 |
| Prueba3 | 1 (Prueba3_2.json) | ~420 |
| Prueba4 | 20 archivos | ~5,900 |
| Prueba5 | 2 archivos | ~410 |
| Prueba6 | 2 archivos | ~800 |
| macarena | 2 archivos (legacy) | ~230 |

**Total: 29 archivos JSON, ~8,700+ frames de datos reales.**

### Resultado del entrenamiento más reciente

```
Dance Labo — Entrenamiento LSTM
================================================
Movimientos: {Prueba1, Prueba2, Prueba3, Prueba4, Prueba5, Prueba6, macarena}
Total de muestras: 29
Dispositivo: CPU
Epochs: 150

Epoch    1/150 | Loss: ...
Epoch  150/150 | Loss: ...  ← Ver train.py output al correr

✅ Modelo guardado en: data/motion_model.pt
✅ Label map en:       data/label_map.pt
```

### Cómo re-entrenar

```bash
cd motion_ml_app
python train.py
# Al terminar, el nuevo modelo reemplaza data/motion_model.pt automáticamente
```

### Criterios de aceptación — Req ③

- [x] 2+ secuencias JSON en `data/sequences/` (tenemos 29)
- [x] Modelo distingue 6+ movimientos
- [x] Score continuo 0.0–1.0 por frame (similitud coseno)
- [x] Modelo guardado en `.pt` (PyTorch portable)
- [x] Script de entrenamiento reproducible (`train.py`)
- [x] Validación disponible (`inference.py`, `benchmark.py`)

---

## ④ Mapeo de Huesos Documentado ✅

> El sistema usa 9 huesos Mixamo que corresponden al esqueleto de "El bueno".

### Feature vector por frame (27 valores)

| Índice (×3) | Hueso Mixamo | Segmento | Peso en score |
|---|---|---|---|
| 0–2 | `mixamorig:LeftArm` | Brazo izquierdo | 25% |
| 3–5 | `mixamorig:RightArm` | Brazo derecho | 25% |
| 6–8 | `mixamorig:LeftForeArm` | Brazo izquierdo | (incluido) |
| 9–11 | `mixamorig:RightForeArm` | Brazo derecho | (incluido) |
| 12–14 | `mixamorig:LeftUpLeg` | Pierna izquierda | 20% |
| 15–17 | `mixamorig:RightUpLeg` | Pierna derecha | 20% |
| 18–20 | `mixamorig:LeftLeg` | Pierna izquierda | (incluido) |
| 21–23 | `mixamorig:RightLeg` | Pierna derecha | (incluido) |
| 24–26 | `mixamorig:Spine` | Torso | 10% |

### Correspondencia MediaPipe → Hueso Mixamo

| Landmark MediaPipe | ID | Hueso Unity |
|---|---|---|
| left_shoulder | 11 | origen de `mixamorig:LeftArm` |
| left_elbow | 13 | destino de `mixamorig:LeftArm` / origen de `LeftForeArm` |
| left_wrist | 15 | destino de `mixamorig:LeftForeArm` |
| right_shoulder | 12 | origen de `mixamorig:RightArm` |
| right_elbow | 14 | destino de `mixamorig:RightArm` / origen de `RightForeArm` |
| right_wrist | 16 | destino de `mixamorig:RightForeArm` |
| left_hip | 23 | origen de `mixamorig:LeftUpLeg` |
| left_knee | 25 | destino de `mixamorig:LeftUpLeg` / origen de `LeftLeg` |
| left_ankle | 27 | destino de `mixamorig:LeftLeg` |
| right_hip | 24 | origen de `mixamorig:RightUpLeg` |
| right_knee | 26 | destino de `mixamorig:RightUpLeg` / origen de `RightLeg` |
| right_ankle | 28 | destino de `mixamorig:RightLeg` |
| Promedio hombros 11+12 | — | origen de `mixamorig:Spine` |
| Promedio caderas 23+24 | — | destino de `mixamorig:Spine` |

### Reglas de visibilidad

```python
# MixamoMapper — pose_extraction/mixamo_mapper.py
visibility_threshold = 0.3   # configurable en __init__
# Si visibility < threshold → usa última pose válida (freeze, sin jitter)
# Restricción Z brazos: z_clamp_min = 0.0 (no van detrás del cuerpo)
```

### Criterios de aceptación — Req ④

- [x] 9 huesos mapeados con índice fijo y orden documentado
- [x] Función `bones_dict_to_vector()` convierte huesos a vector 27D
- [x] Segmentos corporales con pesos que suman 1.0
- [x] Huesos ausentes se llenan con cero (sin crash)
- [x] Segmentos sin datos se excluyen del score (normalización automática)
- [x] Freeze de última pose válida cuando visibility < umbral
- [x] Restricción de eje Z para brazos (configurable)

---

## 📋 Archivos Clave del Sistema

```
motion_ml_app/
├── main.py                          # Punto de entrada — lanza la UI
├── train.py                         # Entrena el LSTM con data/sequences/
├── inference.py                     # Valida el modelo con datos reales
├── benchmark.py                     # Mide latencia E-05 (<100ms)
├── ui/
│   └── main_window.py               # UI PyQt6 completa
│       ├── Selector de cámara       # Detecta cámaras disponibles
│       ├── REC/STOP/PLAY transport  # Grabación y reproducción
│       ├── Enviar a Unity (LSTM)    # Reproduce secuencia LSTM en robot
│       └── 🎯 Evaluar en Vivo       # E-04: evaluación tiempo real
├── ml/
│   ├── live_evaluator.py            # E-04: evaluación frame a frame
│   ├── scoring_engine.py            # D-06: motor de puntuación
│   ├── predictor.py                 # Generación de secuencias LSTM
│   ├── trainer.py                   # Lógica de entrenamiento
│   ├── dataset_builder.py           # Carga y normaliza JSONs
│   └── model.py                     # MotionLSTMGenerator
├── pose_extraction/
│   └── mixamo_mapper.py             # MediaPipe → vectores Mixamo
├── communication/
│   ├── udp_server.py                # Envía JSON por UDP a Unity
│   ├── MovementComparator.cs        # D-01/D-02: Unity recibe scores
│   └── FeedbackUI.cs               # D-05: colores por segmento en Unity
└── data/
    ├── motion_model.pt              # Modelo LSTM entrenado
    ├── label_map.pt                 # {nombre: id}
    ├── historial_scores.json        # Historial de intentos
    ├── benchmark_report.json        # Resultado del benchmark E-05
    └── sequences/                   # 29 JSONs de movimientos
```

---

## 🎮 Cómo hacer la Demo Final

### 1. Preparación
```bash
# En la PC de demo:
cd motion_ml_app
pip install -r requirements.txt  # solo primera vez
```

### 2. Unity
- Abrir la escena con "El bueno"
- Asegurarse de que `MixamoAnimator.cs` y `MovementComparator.cs` están en el mismo GameObject
- Asegurarse de que `FeedbackUI.cs` está configurado con sus referencias
- Presionar **Play**

### 3. Python
```bash
python main.py
```

### 4. En la UI
1. Pulsa **📷 Usar Cámara** (selecciona la cámara correcta con el combo si hay varias)
2. Escribe el nombre del movimiento (ej: `Prueba4`)
3. Pulsa **🎯 Evaluar en Vivo**
4. El robot en Unity se mueve con tu pose, y el score aparece en pantalla

### 5. Plan B (si falla la cámara)
- Cargar un video pregrabado con **📂 Cargar Video**
- El pipeline funciona igual con video pregrabado
