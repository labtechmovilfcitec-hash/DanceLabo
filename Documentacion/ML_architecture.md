Arquitectura de Flow Training
## Dance Labo · E-01 · Sprint 6

---

## 1. Problema a Resolver

El sistema necesita comparar la secuencia de movimiento de un estudiante (MediaPipe/MixamoMapper live) contra una secuencia ideal generada por la IA, con tolerancia a diferencias de velocidad y fisionomía corporal.

**Input de Generación:** ID del movimiento (embedding de clase)  
**Output de Generación:** Secuencia de poses ideales tridimensionales de 9 huesos Mixamo (27 valores por frame)  
**Scoring:** Puntuación de similitud coseno frame a frame (0.0–1.0) por segmento corporal

---

## 2. Alternativas Evaluadas

| Enfoque | Ventajas | Desventajas | Decisión |
|---|---|---|---|
| **DTW** | Simple, sin entrenamiento, tolerante a velocidad | No aprende, no generaliza, sensible a ruido | Descartado |
| **HMM** | Modelo probabilístico de secuencias | Requiere muchos datos, difícil en 27 dimensiones | Descartado |
| **Similitud coseno** | Muy rápida | Sin memoria temporal, ignora contexto | Solo se usa en scoring interno |
| **LSTM Generativo** | Aprende patrón temporal, genera secuencia ideal, generaliza con pocos datos | Requiere entrenamiento | **Elegido** |

---

## 3. Arquitectura: LSTM Generativo

### Justificación

El modelo aprende a **generar** la secuencia ideal dado un ID de movimiento. Ventajas sobre comparación directa (DTW):
1. Comparación en cualquier frame sin alinear secuencias complejas
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
  [Linear → 27]          ← proyecta a rotaciones/vectores de los 9 huesos
        │
        ▼
Secuencia generada (T × 27)
        │
        ├── vs Pose real del estudiante (T × 27)
        ▼
  [ScoringEngine]        ← similitud coseno por segmento
```

### Parámetros (ml/model.py y train.py)

```python
num_classes    = len(dataset.label_map)  # Dinámico según secuencias (ej: 2 clases)
hidden_size    = 256
num_layers     = 3
output_size    = 27    # 9 huesos Mixamo × XYZ
max_seq_length = dataset.max_length      # Dinámico según la secuencia más larga (~150+ frames)
```

---

## 4. Features por Frame

Vector de **27 valores**: `[bone0.x, bone0.y, bone0.z, ..., bone8.x, bone8.y, bone8.z]`

Los huesos mapeados y su orden en el vector son:
1. `mixamorig:LeftArm`
2. `mixamorig:RightArm`
3. `mixamorig:LeftForeArm`
4. `mixamorig:RightForeArm`
5. `mixamorig:LeftUpLeg`
6. `mixamorig:RightUpLeg`
7. `mixamorig:LeftLeg`
8. `mixamorig:RightLeg`
9. `mixamorig:Spine`

Estos valores son calculados por `MixamoMapper` a partir de los landmarks de MediaPipe y corresponden a vectores unitarios de dirección 3D, lo que aísla la evaluación de la escala y altura del usuario.

---

## 5. Dataset

Formato JSON en `data/sequences/`:
```json
{
  "movement_name": "Secuencia1",
  "frames": [
    {
      "frame": 0,
      "landmarks": {
        "mixamorig:LeftArm": {"x": 0.54, "y": -0.12, "z": -0.83},
        "mixamorig:RightArm": {"x": -0.52, "y": -0.15, "z": -0.82},
        ...
      }
    }
  ]
}
```
`MotionDataset` (dataset_builder.py) carga, estructura y aplica padding automático para los batches de entrenamiento.

---

## 6. Métricas

| Fase | Métrica | Objetivo |
|---|---|---|
| Entrenamiento | MSE entre secuencia generada y real | Loss < 0.01 |
| Inferencia | Similitud coseno por segmento | Score > 0.80 en pruebas de réplica |

---

## 7. Estado de Implementación

| Componente | Archivo | Estado |
|---|---|---|
| MotionLSTMGenerator | `ml/model.py` | Implementado |
| MotionDataset | `ml/dataset_builder.py` | Implementado |
| ScoringEngine | `ml/scoring_engine.py` | Implementado |
| Script de entrenamiento | `train.py` | Implementado |
| Secuencias grabadas | `data/sequences/*.json` | Implementado (32 secuencias grabadas) |
| Modelo entrenado | `data/motion_model.pt` | Implementado (Modelo entrenado) |

---

**Revisado por el equipo:** Dance Labo Core Team  Fecha: Mayo 2026
