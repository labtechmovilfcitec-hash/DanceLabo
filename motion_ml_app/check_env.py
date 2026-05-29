"""
check_env.py — Verificación del Entorno Python
Dance Labo · A-01

Uso:
    cd motion_ml_app
    python check_env.py

Verifica que todas las dependencias están instaladas y que el entorno
es compatible con el sistema antes de ejecutar main.py.
"""

import sys
import importlib
import platform

# Fix encoding para terminales Windows cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Configuración ────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("PyQt6",          "PyQt6",           "Interfaz gráfica principal"),
    ("cv2",            "opencv-python",   "Captura y procesamiento de video"),
    ("ultralytics",    "ultralytics",     "YOLO — detección de persona"),
    ("mediapipe",      "mediapipe",       "Estimación de pose 3D (33 landmarks)"),
    ("torch",          "torch",           "PyTorch — entrenamiento y carga del modelo LSTM"),
    ("sklearn",        "scikit-learn",    "Normalización de features (StandardScaler)"),
    ("numpy",          "numpy",           "Álgebra vectorial"),
    ("filterpy",       "filterpy",        "Filtro de Kalman (suavizado de datos)"),
    ("json",           None,              "Estándar — serialización JSON"),
    ("socket",         None,              "Estándar — comunicación UDP"),
    ("threading",      None,              "Estándar — hilos"),
]

MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 12)

# ─── Helpers ──────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠  {msg}{RESET}")
def fail(msg):  print(f"  {RED}❌ {msg}{RESET}")


# ─── Verificaciones ────────────────────────────────────────────────────────────

def check_python_version():
    print(f"\n{BOLD}🐍  Python{RESET}")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON:
        fail(f"Python {version_str} — requiere >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
        return False
    if (v.major, v.minor) > MAX_PYTHON:
        warn(f"Python {version_str} — versión más nueva de lo probado ({MAX_PYTHON[0]}.{MAX_PYTHON[1]}). "
             "Puede funcionar, pero verifica compatibilidad de MediaPipe.")
        return True
    ok(f"Python {version_str} — versión compatible")
    return True


def check_packages():
    print(f"\n{BOLD}📦  Paquetes requeridos{RESET}")
    all_ok = True
    for import_name, pip_name, description in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            ok(f"{import_name:<18} v{version:<12} — {description}")
        except ImportError:
            if pip_name:
                fail(f"{import_name:<18} NO ENCONTRADO — instalar con:  pip install {pip_name}")
            else:
                fail(f"{import_name:<18} NO ENCONTRADO (módulo estándar — ¿Python corrompido?)")
            all_ok = False
    return all_ok


def check_model_files():
    import os
    print(f"\n{BOLD}🤖  Archivos del modelo ML{RESET}")
    base = os.path.dirname(os.path.abspath(__file__))
    files_to_check = {
        "data/motion_model.pt": "Modelo LSTM entrenado",
        "data/label_map.pt":    "Mapa de etiquetas (movimientos conocidos)",
        "data/sequences":       "Carpeta de secuencias de referencia",
    }
    all_ok = True
    for rel_path, desc in files_to_check.items():
        full = os.path.join(base, rel_path)
        if os.path.exists(full):
            size_kb = os.path.getsize(full) // 1024 if os.path.isfile(full) else -1
            size_str = f" ({size_kb} KB)" if size_kb >= 0 else ""
            ok(f"{rel_path}{size_str} — {desc}")
        else:
            fail(f"{rel_path} — {desc} (ejecuta python train.py para generarlo)")
            all_ok = False
    return all_ok


def check_camera():
    print(f"\n{BOLD}📷  Cámaras{RESET}")
    try:
        import cv2
        found = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                found.append(i)
                cap.release()
        if found:
            ok(f"Cámaras detectadas: índices {found}")
        else:
            warn("No se detectaron cámaras con DirectShow. "
                 "Prueba conectar una webcam y volver a ejecutar.")
    except Exception as e:
        warn(f"Error detectando cámaras: {e}")


def check_yolo_weights():
    import os
    print(f"\n{BOLD}🎯  Pesos YOLO{RESET}")
    base = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.join(base, "yolov8n.pt")
    if os.path.isfile(yolo_path):
        size_kb = os.path.getsize(yolo_path) // 1024
        ok(f"yolov8n.pt ({size_kb} KB) — encontrado")
    else:
        warn("yolov8n.pt no encontrado — se descargará automáticamente la primera vez que inicies la app.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Dance Labo — Verificación de Entorno (A-01)")
    print(f"  Sistema: {platform.system()} {platform.release()}")
    print("=" * 60)

    results = [
        check_python_version(),
        check_packages(),
        check_model_files(),
    ]
    check_camera()
    check_yolo_weights()

    print("\n" + "─" * 60)
    if all(results):
        print(f"  {GREEN}{BOLD}✅  Entorno listo — puedes ejecutar: python main.py{RESET}")
    else:
        print(f"  {RED}{BOLD}❌  Hay errores — revisa los puntos marcados arriba.{RESET}")
        sys.exit(1)
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
