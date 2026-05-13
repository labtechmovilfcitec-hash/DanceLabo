# validacion_final.md — Validación de los 4 Requerimientos Mínimos
## Dance Labo · F-04 · Sprint 6
**Última actualización:** Mayo 2026  
**Sistema:** Dance Labo — LSTM sobre huesos Mixamo (X Bot / El bueno)

---

## Checklist General

| # | Requerimiento | Estado |
|---|---|---|
| ① | Detecta réplica y muestra feedback visual | ⏳ Lógica lista — pendiente video con pipeline completo |
| ② | Robot 3D mejorado (rig + aprende movimientos) | ⏳ LSTM entrenado — pendiente mejora del rig en P3 |
| ③ | 2+ secuencias grabadas y modelo entrenado | ✅ **COMPLETO** |
| ④ | Mapeo de huesos documentado | ✅ **COMPLETO** |

---

## ③ Secuencias Grabadas y Modelo Entrenado ✅

> El modelo LSTM fue entrenado con secuencias reales grabadas desde Unity.

### Secuencias disponibles (`data/sequences/`)

| Archivo | Movimiento | Frames | Huesos con datos |
|---|---|---|---|
| `macarena_0.json` | macarena | 191 | LeftArm, RightArm, ForeArms, Spine |
| `Nose_1.json` | Nose | 88 | LeftArm, RightArm, ForeArms, Spine |
| `Nose1_2.json` | Nose1 | 111 | LeftArm, RightArm, ForeArms, Spine |
| `Nose2_3.json` | Nose2 | 91 | LeftArm, RightArm, ForeArms, Spine |
| `Nose3_4.json` | Nose3 | 70 | LeftArm, RightArm, ForeArms, Spine |

**Total: 5 movimientos distintos, 551 frames de datos reales.**

### Resultado del entrenamiento (`python train.py`)

```
Movimientos: {'macarena': 0, 'Nose1': 1, 'Nose2': 2, 'Nose3': 3, 'Nose': 4}
Total de muestras:      5
Frames máx. secuencia:  191
Dispositivo:            CPU
Epochs:                 150

Epoch    1/150 | Loss: 0.126715
Epoch  100/150 | Loss: 0.042068
Epoch  150/150 | Loss: 0.038878  ← Loss final

✅ Modelo guardado en: data/motion_model.pt
✅ Label map en:       data/label_map.pt
   Mejora total:       ~69% reducción de error
```

### Validación del modelo (`python inference.py`)

Similitud coseno entre secuencia generada por LSTM y secuencia real:

| Movimiento | Similitud bruta | Brazo izq. | Brazo der. | Torso |
|---|---|---|---|---|
| macarena | 93.6% | 🟢 92.8% | 🟢 92.5% | 🟢 99.2% |
| Nose1 | 83.5% | 🟡 77.1% | 🟡 71.1% | 🟢 98.3% |
| Nose2 | 88.0% | 🟢 89.8% | 🟡 78.8% | 🟢 98.9% |
| Nose3 | 86.6% | 🟢 89.9% | 🟡 73.5% | 🟢 99.3% |
| Nose | 89.7% | 🟢 87.4% | 🟢 85.2% | 🟢 96.5% |

> **Nota:** Las piernas no se puntúan porque los JSONs grabados por P3 no incluían huesos de pierna. El scoring se normaliza automáticamente sobre los segmentos con datos. Cuando P3 grabe secuencias con piernas, el modelo se re-entrena automáticamente con `python train.py`.

### Archivos generados
```
data/
├── motion_model.pt    ← pesos del modelo entrenado
├── label_map.pt       ← mapeo {nombre_movimiento: id}
└── sequences/
    ├── macarena_0.json
    ├── Nose_1.json
    ├── Nose1_2.json
    ├── Nose2_3.json
    └── Nose3_4.json
```

### Criterios de aceptación — Req ③

- [x] Secuencias JSON existen en `data/sequences/`
- [x] Modelo distingue 5 movimientos distintos
- [x] Score continuo 0.0–1.0 por frame (similitud coseno)
- [x] Modelo guardado en `.pt` (PyTorch portable)
- [x] Script de entrenamiento reproducible (`train.py`)
- [x] Validación de precisión documentada (`inference.py`)
- [ ] Re-entrenar cuando P3 añada más secuencias con piernas

---

## ④ Mapeo de Huesos del Sistema ML ✅

> El sistema ML usa 9 huesos Mixamo del esqueleto de X Bot / El bueno.

### Decisión de diseño

El sistema **NO usa los 33 landmarks de MediaPipe** directamente. Usa los **vectores de dirección de los huesos del robot 3D en Unity**, grabados por `PoseRecorder.cs`. Esto es más preciso para evaluar la pose 3D del robot porque:

1. Los landmarks de MediaPipe son posiciones en espacio de cámara (2.5D)
2. Los huesos del robot son rotaciones reales en espacio 3D
3. La comparación se hace directamente en el dominio del robot → mayor coherencia

### Huesos Mixamo utilizados (feature vector de 27 valores)

| Índice | Hueso Mixamo | Segmento | Peso en score |
|---|---|---|---|
| 0 | `mixamorig:LeftArm` | Brazo izquierdo | 25% |
| 1 | `mixamorig:RightArm` | Brazo derecho | 25% |
| 2 | `mixamorig:LeftForeArm` | Brazo izquierdo | (incluido arriba) |
| 3 | `mixamorig:RightForeArm` | Brazo derecho | (incluido arriba) |
| 4 | `mixamorig:LeftUpLeg` | Pierna izquierda | 20% |
| 5 | `mixamorig:RightUpLeg` | Pierna derecha | 20% |
| 6 | `mixamorig:LeftLeg` | Pierna izquierda | (incluido arriba) |
| 7 | `mixamorig:RightLeg` | Pierna derecha | (incluido arriba) |
| 8 | `mixamorig:Spine` | Torso | 10% |

**Vector por frame:** `[LeftArm.x, LeftArm.y, LeftArm.z, RightArm.x, ..., Spine.z]` → 27 valores

### Algoritmo de comparación

```
Pose estudiante (huesos Mixamo vía UDP desde Unity)
    → bones_dict_to_vector()  → vector de 27 floats
                    ↕ cosine_similarity()
Pose referencia (LSTM generada para el movimiento)
    → lstm_frame_to_vector()  → vector de 27 floats

Resultado por segmento → color verde/amarillo/rojo → FeedbackUI
```

### Criterios de aceptación — Req ④

- [x] 9 huesos mapeados con índice fijo y orden documentado
- [x] Función `bones_dict_to_vector()` convierte huesos a vector normalizado
- [x] Segmentos corporales definidos con pesos que suman 1.0
- [x] Huesos ausentes se llenan con cero (sin crash)
- [x] Segmentos sin datos se excluyen automáticamente del score

---

## ① Detecta Réplica ⏳

> El sistema detecta cuando el estudiante replica el movimiento y muestra retroalimentación visual.

**Lógica implementada (P4):**
```
ScoringEngine.evaluate_frame(bones_dict, lstm_reference)
→ {"overall": 0.87, "overall_color": "verde", "segments": {...}}
```

**Pendiente para completar:**
- [ ] P5 debe conectar `FeedbackUI.cs` a los scores del `ScoringEngine`
- [ ] P2 debe tener el bridge Python enviando huesos en vivo
- [ ] P1 debe tener `StudentReceiver.cs` recibiendo datos
- [ ] Grabar video de demostración con persona real frente a cámara

---

## ② Robot 3D Mejorado ⏳

> El modelo 3D de Labo tiene un rig mejorado y el sistema aprende su movimiento.

**Implementado (P4):**
- LSTM entrenado y generando secuencias de referencia ✅
- Dataset builder lee automáticamente nuevos JSONs ✅

**Pendiente para completar:**
- [ ] P3 debe mejorar el rig del esqueleto (más huesos mapeados, incluyendo piernas)
- [ ] P3 debe grabar nuevas secuencias con el rig mejorado
- [ ] Re-entrenar el modelo con `python train.py` al tener nuevos JSONs

---

## 📋 Resumen de Pendientes por Equipo

| Pendiente | Responsable | Bloqueado en |
|---|---|---|
| Mejorar rig del modelo 3D | **P3** | Req ② completo |
| Grabar secuencias con piernas | **P3** | Modelo con datos de piernas |
| Bridge Python corriendo en vivo | **P2** | Req ① completo |
| `StudentReceiver.cs` recibiendo datos | **P1** | Req ① completo |
| `FeedbackUI.cs` conectada al scoring | **P5** | Req ① completo |
| Video de demostración | **P4 + P5** | Todo lo anterior |
| Re-entrenar con nuevas secuencias | **P4** | Esperar JSONs de P3 |
