# ml_architecture.md — Arquitectura de Flow Training
## Dance Labo · E-01 · Sprint 1

---

## 1. Problema a Resolver

El sistema necesita comparar la secuencia de movimiento de un estudiante (MediaPipe live) contra una secuencia aprendida, con tolerancia a diferencias de velocidad.

**Input:** 33 landmarks (x,y,z) por frame → vector de 99 valores + ID del movimiento  
**Output:** Secuencia de referencia generada + puntuación de similitud frame a frame (0.0–1.0)

---

## 2. Alternativas Evaluadas

| Enfoque | Ventajas | Desventajas | Decisión |
|---|---|---|---|
| **DTW** | Simple, sin entrenamiento, tolerante a velocidad | No aprende, no generaliza, sensible a ruido | ❌ Descartado |
| **HMM** | Modelo probabilístico de secuencias | Requiere muchos datos, difícil en 99 dimensiones | ❌ Descartado |
| **Similitud coseno** | Muy rápida | Sin memoria temporal, ignora contexto | ❌ Solo se usa en scoring interno |
| **LSTM Generativo** | Aprende patrón temporal, genera secuencia ideal, generaliza con pocos datos | Requiere entrenamiento | ✅ **Elegido** |

---

## 3. Arquitectura: LSTM Generativo

### Justificación

El modelo aprende a **generar** la secuencia ideal dado un ID de movimiento. Ventajas sobre comparación directa (DTW):
1. Comparación en cualquier frame sin alinear secuencias
2. Longitud ajustable (tolerancia de velocidad)
3. Generaliza con pocas muestras de entrenamiento

### Diagrama

```
ID de Movimiento (int)
        │
        ▼
  [Embedding 256d]       ← aprende la "firma" de cada movimiento
        │
        ▼
  [LSTM 3 capas]         ← extiende la firma a lo largo del tiempo
  hidden_size=256
        │
        ▼
  [Linear → 99]          ← proyecta a coordenadas 3D de landmarks
        │
        ▼
Secuencia generada (T × 99)
        │
        ├── vs Pose real del estudiante (T × 99)
        ▼
  [ScoringEngine]        ← similitud coseno por segmento
```

### Parámetros (model.py)

```python
num_classes    = 2     # secuencia_01 y secuencia_02
hidden_size    = 256
num_layers     = 3
output_size    = 99    # 33 landmarks × xyz
max_seq_length = 150   # ~5 segundos a 30 FPS
```

---

## 4. Features por Frame

Vector de **99 valores**: `[lm0.x, lm0.y, lm0.z, lm1.x, ..., lm32.z]`

Los landmarks provienen de `MediaPipe pose_world_landmarks` (coordenadas 3D en metros relativos a la cadera — no requieren normalización adicional).

---

## 5. Dataset

Formato JSON en `data/sequences/`:
```json
{
  "movement_name": "secuencia_01",
  "frames": [
    {"frame": 0, "landmarks": {"0": {"x": 0.02, "y": -0.45, "z": -0.01}, ...}}
  ]
}
```
`MotionDataset` (dataset_builder.py) carga y aplica padding automático para batches.

---

## 6. Métricas

| Fase | Métrica | Objetivo |
|---|---|---|
| Entrenamiento | MSE entre secuencia generada y real | Loss < 0.01 |
| Inferencia | Similitud coseno por segmento | Score > 0.80 en prueba |

---

## 7. Estado de Implementación

| Componente | Archivo | Estado |
|---|---|---|
| MotionLSTMGenerator | `ml/model.py` | ✅ Implementado |
| MotionDataset | `ml/dataset_builder.py` | ✅ Implementado |
| ScoringEngine | `ml/scoring_engine.py` | ✅ Implementado |
| Script de entrenamiento | `train.py` | ✅ Implementado |
| Secuencias grabadas | `data/sequences/*.json` | ⏳ Pendiente grabación |
| Modelo entrenado | `data/motion_model.pt` | ⏳ Pendiente entrenamiento |

---

**Revisado por el equipo:** _______________________ Fecha: ___________
