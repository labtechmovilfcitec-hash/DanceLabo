"""
inference.py — Validación del Modelo LSTM Entrenado
Dance Labo · F-04

Uso:
    cd motion_ml_app
    python inference.py
"""

import os
import json
import torch
import numpy as np

from ml.model import MotionLSTMGenerator
from ml.scoring_engine import (
    ScoringEngine, bones_dict_to_vector,
    lstm_frame_to_vector, cosine_similarity,
    EXPECTED_BONES
)

MODEL_PATH = "../data/motion_model.pt"
LABEL_PATH = "../data/label_map.pt"
SEQ_DIR    = "../data/sequences"


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No se encontró el modelo en '{MODEL_PATH}'")
        print("  Ejecuta primero: python train.py")
        return None, None

    label_map   = torch.load(LABEL_PATH)
    model       = MotionLSTMGenerator(num_classes=len(label_map), output_size=27)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print(f"✅ Modelo cargado — movimientos conocidos:")
    for name, idx in label_map.items():
        print(f"   [{idx}] {name}")
    return model, label_map


def load_real_sequence(movement_name):
    for fname in os.listdir(SEQ_DIR):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(SEQ_DIR, fname)) as f:
            data = json.load(f)
        if data.get("movement_name") == movement_name:
            return [
                bones_dict_to_vector(frame.get("landmarks", {}))
                for frame in data.get("frames", [])
            ]
    return []


def run():
    print("=" * 55)
    print("  Dance Labo — Inferencia y Validación del Modelo")
    print("=" * 55 + "\n")

    model, label_map = load_model()
    if model is None:
        return

    summary = []

    for movement, label_id in label_map.items():
        print(f"{'─'*45}")
        print(f"🎯  Movimiento: '{movement}'  (ID={label_id})")

        # Generar secuencia de referencia con el LSTM
        with torch.no_grad():
            generated = model(torch.tensor([label_id]))  # (1, T, 27)
        generated_np = generated[0].numpy()              # (T, 27)

        # Cargar secuencia real
        real_vecs = load_real_sequence(movement)
        if not real_vecs:
            print("  ⚠️  No hay JSON para este movimiento — saltando\n")
            continue

        n = min(len(generated_np), len(real_vecs))
        print(f"  Frames comparados: {n}  "
              f"(LSTM={len(generated_np)}, real={len(real_vecs)})\n")

        # Comparar frame a frame con ScoringEngine
        engine = ScoringEngine()
        engine.start_session(movement)

        frame_sims = []
        for i in range(n):
            bones_dict = {
                bone: {
                    'x': float(real_vecs[i][bi * 3]),
                    'y': float(real_vecs[i][bi * 3 + 1]),
                    'z': float(real_vecs[i][bi * 3 + 2]),
                }
                for bi, bone in enumerate(EXPECTED_BONES)
            }
            engine.evaluate_frame(bones_dict, generated_np[i])
            frame_sims.append(
                cosine_similarity(real_vecs[i], lstm_frame_to_vector(generated_np[i]))
            )

        final = engine.end_session()

        print(f"  Similitud coseno promedio: {np.mean(frame_sims)*100:.1f}%")
        print(f"  Score por segmento:")
        for seg, pct in final['desglose'].items():
            icon = "🟢" if pct >= 85 else ("🟡" if pct >= 60 else "🔴")
            bar  = "█" * int(pct / 5)
            print(f"    {icon} {seg:<22} {pct:5.1f}%  {bar}")

        pct = final['score_pct']
        if pct >= 85:
            verdict = "✅ El modelo aprendió este movimiento MUY BIEN"
        elif pct >= 60:
            verdict = "⚠️  El modelo aprendió parcialmente"
        else:
            verdict = "❌ Necesita más datos o epochs"

        print(f"\n  {verdict}")
        print(f"  Score final: {pct}% — {final['label']}\n")
        summary.append((movement, pct, final['label']))

    # Resumen
    print("=" * 55)
    print("  RESUMEN")
    print("=" * 55)
    for mov, pct, label in summary:
        icon = "✅" if pct >= 85 else ("⚠️ " if pct >= 60 else "❌")
        print(f"  {icon} {mov:<20} {pct:5.1f}%  — {label}")

    if summary:
        avg = np.mean([r[1] for r in summary])
        print(f"\n  Promedio general: {avg:.1f}%")
        msg = "🎉 Modelo listo para producción." if avg >= 75 else "🔁 Recomienda más secuencias."
        print(f"  {msg}")


if __name__ == "__main__":
    run()
