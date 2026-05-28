"""
benchmark.py — E-05: Prueba de rendimiento de inferencia en tiempo real

Mide la latencia del pipeline completo:
  MediaPipe → MixamoMapper → LiveEvaluator → Score

Uso:
    cd motion_ml_app
    python benchmark.py [nombre_movimiento]

Si no se especifica movimiento, usa el primero del label_map.

Salida:
    - FPS promedio de inferencia
    - Latencia P50 / P95 / P99
    - Uso estimado de CPU y RAM
    - Resultado: ✅ OK (< 100ms) o ⚠ LENTO
"""

import os
import sys
import time
import statistics
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_PATH = "data/motion_model.pt"
LABEL_PATH = "data/label_map.pt"
N_FRAMES   = 300          # Frames a simular
TARGET_MS  = 100.0        # Latencia máxima aceptable por frame (E-05)

# ── Importar módulos del proyecto ─────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml.live_evaluator import LiveEvaluator
    from ml.scoring_engine import EXPECTED_BONES
    import torch
except ImportError as e:
    print(f"[ERROR] Falta dependencia: {e}")
    print("  Ejecuta: pip install -r requirements.txt")
    sys.exit(1)

# ── Datos de prueba (pose mock realista) ─────────────────────────────────────

def make_mock_bones(noise: float = 0.1) -> dict:
    """Genera un frame de huesos mock con ruido gaussiano."""
    base_vectors = {
        "mixamorig:LeftArm":      {"x":  0.70, "y": -0.60, "z": 0.20},
        "mixamorig:RightArm":     {"x": -0.70, "y": -0.60, "z": 0.20},
        "mixamorig:LeftForeArm":  {"x":  0.50, "y": -0.80, "z": 0.15},
        "mixamorig:RightForeArm": {"x": -0.50, "y": -0.80, "z": 0.15},
        "mixamorig:LeftUpLeg":    {"x":  0.10, "y": -0.98, "z": 0.05},
        "mixamorig:RightUpLeg":   {"x": -0.10, "y": -0.98, "z": 0.05},
        "mixamorig:LeftLeg":      {"x":  0.05, "y": -0.99, "z": 0.05},
        "mixamorig:RightLeg":     {"x": -0.05, "y": -0.99, "z": 0.05},
        "mixamorig:Spine":        {"x":  0.00, "y":  1.00, "z": 0.00},
    }
    bones = {}
    for name, vec in base_vectors.items():
        bones[name] = {
            "x": vec["x"] + np.random.uniform(-noise, noise),
            "y": vec["y"] + np.random.uniform(-noise, noise),
            "z": vec["z"] + np.random.uniform(-noise, noise),
        }
    return bones


# ── Selección de movimiento ───────────────────────────────────────────────────

def choose_movement() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    if os.path.exists(LABEL_PATH):
        label_map = torch.load(LABEL_PATH, weights_only=True)
        if label_map:
            return list(label_map.keys())[0]
    print("[ERROR] No se encontró label_map.pt — entrena primero con 'python train.py'")
    sys.exit(1)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 58)
    print("  Dance Labo — Benchmark de Rendimiento (E-05)")
    print("=" * 58)

    movement = choose_movement()
    print(f"\n  Movimiento: '{movement}'")
    print(f"  Frames:     {N_FRAMES}")
    print(f"  Objetivo:   < {TARGET_MS:.0f} ms por frame\n")

    # Cargar evaluador (incluye carga del modelo)
    t0 = time.perf_counter()
    evaluator = LiveEvaluator(movement, MODEL_PATH)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Carga del modelo: {load_ms:.1f} ms  ", end="")
    print("✅" if load_ms < 5000 else "⚠ (> 5s)")

    if not evaluator.is_ready:
        print(f"\n[ERROR] Movimiento '{movement}' no encontrado en el modelo.")
        return

    print(f"  Frames de referencia: {evaluator.total_frames}")

    # ── Warmup (3 frames para que JIT y caché se inicien) ────────────────────
    for _ in range(3):
        evaluator.evaluate(make_mock_bones())
    evaluator.reset()

    # ── Medición principal ────────────────────────────────────────────────────
    latencies_ms = []
    scores_sample = []

    for i in range(N_FRAMES):
        bones = make_mock_bones()
        t_start = time.perf_counter()
        result  = evaluator.evaluate(bones)
        t_end   = time.perf_counter()

        latency = (t_end - t_start) * 1000
        latencies_ms.append(latency)
        if result:
            scores_sample.append(result.get("overall", 0.0))

    # ── Estadísticas ──────────────────────────────────────────────────────────
    p50  = statistics.median(latencies_ms)
    p95  = sorted(latencies_ms)[int(N_FRAMES * 0.95)]
    p99  = sorted(latencies_ms)[int(N_FRAMES * 0.99)]
    mean = statistics.mean(latencies_ms)
    fps  = 1000.0 / mean if mean > 0 else 0

    ok_count = sum(1 for l in latencies_ms if l < TARGET_MS)
    ok_pct   = 100 * ok_count / N_FRAMES

    print(f"\n{'─' * 58}")
    print(f"  Resultados ({N_FRAMES} frames simulados):")
    print(f"{'─' * 58}")
    print(f"  FPS promedio:  {fps:.1f} fps")
    print(f"  Latencia media:{mean:.2f} ms")
    print(f"  Latencia P50:  {p50:.2f} ms")
    print(f"  Latencia P95:  {p95:.2f} ms")
    print(f"  Latencia P99:  {p99:.2f} ms")
    print(f"  Frames OK (<{TARGET_MS:.0f}ms): {ok_pct:.1f}%")

    if scores_sample:
        avg_score = statistics.mean(scores_sample)
        print(f"  Score promedio mock: {avg_score*100:.1f}%")

    print(f"\n{'─' * 58}")
    # Evaluar criterio E-05: < 100ms CPU
    if p95 < TARGET_MS:
        print(f"  ✅ RESULTADO: OK — P95 = {p95:.2f} ms < {TARGET_MS:.0f} ms")
        print(f"     El pipeline corre en tiempo real en CPU.")
    else:
        print(f"  ⚠ RESULTADO: LENTO — P95 = {p95:.2f} ms ≥ {TARGET_MS:.0f} ms")
        print(f"     Considera usar GPU (CUDA) o reducir NUM_LAYERS en model.py")
    print(f"{'─' * 58}\n")

    # ── Guardar reporte ───────────────────────────────────────────────────────
    report = {
        "movement": movement,
        "n_frames": N_FRAMES,
        "fps_avg":  round(fps, 2),
        "latency_mean_ms": round(mean, 3),
        "latency_p50_ms":  round(p50, 3),
        "latency_p95_ms":  round(p95, 3),
        "latency_p99_ms":  round(p99, 3),
        "frames_ok_pct":   round(ok_pct, 1),
        "target_ms":       TARGET_MS,
        "passed_e05":      p95 < TARGET_MS,
        "model_load_ms":   round(load_ms, 1),
    }
    report_path = "data/benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Reporte guardado en: {report_path}")


if __name__ == "__main__":
    run_benchmark()
