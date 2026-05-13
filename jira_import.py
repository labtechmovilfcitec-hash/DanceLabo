"""
Script para importar issues a Jira - Dance Labo
Corre esto en tu computadora:
  python jira_import.py
"""
import requests
import json
import time
from requests.auth import HTTPBasicAuth

# ── Configuración ──────────────────────────────────────────────
JIRA_DOMAIN = "uabc-team-lic4oydp.atlassian.net"
PROJECT_KEY = "LAB"
EMAIL       = "daniel.tornero@uabc.edu.mx"
API_TOKEN   = "AQUI_TU_TOKEN_DE_JIRA"
# ───────────────────────────────────────────────────────────────

BASE_URL  = f"https://{JIRA_DOMAIN}/rest/api/3"
AGILE_URL = f"https://{JIRA_DOMAIN}/rest/agile/1.0"
auth      = HTTPBasicAuth(EMAIL, API_TOKEN)
headers   = {"Accept": "application/json", "Content-Type": "application/json"}

SPRINTS = {
    1: {"name": "Sprint 1 — Infraestructura base",    "start": "2026-04-10", "end": "2026-04-17"},
    2: {"name": "Sprint 2 — Comunicacion validada",   "start": "2026-04-17", "end": "2026-04-24"},
    3: {"name": "Sprint 3 — Secuencias grabadas",     "start": "2026-04-24", "end": "2026-05-05"},
    4: {"name": "Sprint 4-5 — Comparador y ML",       "start": "2026-05-05", "end": "2026-05-14"},
    5: {"name": "Sprint 6 — ML integrado end-to-end", "start": "2026-05-14", "end": "2026-05-20"},
    6: {"name": "Sprint 7 — Demo final",              "start": "2026-05-20", "end": "2026-05-27"},
}

ISSUES = [
    # ── Sprint 1 ───────────────────────────────────────────────
    {
        "summary": "[A-01] Configurar entorno Python (YOLO + MediaPipe + OSC) — P2",
        "priority": "High", "labels": ["P2-Python"], "sprint": 1,
        "description": "Configurar entorno Python con todas las dependencias del proyecto.",
        "ac": ["Crea requirements.txt con dependencias fijas (ultralytics, mediapipe==0.10.x, python-osc, opencv-python, numpy)", "Entorno funciona en Python 3.9-3.11", "Script check_env.py confirma imports sin errores", "Probado en Windows", "README con instrucciones de instalacion desde cero"]
    },
    {
        "summary": "[A-02] Bridge Python yolo_mediapipe_bridge.py — P2",
        "priority": "High", "labels": ["P2-Python"], "sprint": 1,
        "description": "Desarrollar el bridge Python que usa YOLO para detectar bounding box y MediaPipe para estimar pose 3D. Envia datos por OSC a Unity.",
        "ac": ["YOLO detecta bounding box de la persona", "MediaPipe estima pose 3D dentro del recorte de YOLO", "Convierte pose_world_landmarks a rotaciones (Quaternion por par parent-child)", "Envia /Student/Bone/Pos a 127.0.0.1:39540", "Envia /Student/Visibility con confianza por parte del cuerpo", "Corre a minimo 15 FPS en CPU (i5 o equivalente)", "Ventana de preview con esqueleto dibujado para debug", "Cierre limpio con Q o Ctrl+C"]
    },
    {
        "summary": "[A-03] Crear StudentReceiver.cs en Unity — P1",
        "priority": "High", "labels": ["P1-Unity"], "sprint": 1,
        "description": "Script C# que escucha en puerto 39540 y recibe la pose del estudiante enviada desde Python por OSC.",
        "ac": ["Escucha en puerto 39540 (separado del 39539 de Dollars Mono)", "Recibe /Student/Bone/Pos con [boneName, px, py, pz, rx, ry, rz, rw]", "Recibe /Student/Visibility con confianza por parte del cuerpo", "Almacena en diccionarios studentBonePositions y studentBoneRotations", "Expone IsStudentDetected (bool) y DetectionConfidence (float 0-1)", "Sin errores si el bridge Python no esta activo", "Namespace Dollars, mismo patron que ExternalReceiver.cs"]
    },
    {
        "summary": "[B-01] Rig del modelo robot — revisar y mejorar Humanoid rig — P3",
        "priority": "High", "labels": ["P3-3D"], "sprint": 1,
        "description": "Revisar y mejorar el rig Humanoid del robot.fbx en Unity para que los datos de Dollars Mono se apliquen sin deformaciones.",
        "ac": ["Avatar Humanoid valido sin huesos en rojo en Unity", "Huesos principales mapeados: columna, cabeza, hombros, brazos, codos, munecas, caderas, rodillas, tobillos", "Huesos de dedos mapeados si el modelo los tiene", "Sin deformaciones visuales al recibir datos de Dollars Mono", "Avatar pasa validacion de Unity sin advertencias", "Documentado que huesos quedaron sin mapear y por que"]
    },
    {
        "summary": "[E-01] Definir arquitectura de Flow Training — P4",
        "priority": "High", "labels": ["P4-ML"], "sprint": 1,
        "description": "Investigar y documentar la arquitectura del modelo ML para comparar secuencias de movimiento con tolerancia a diferencias de velocidad.",
        "ac": ["Investiga y documenta al menos 3 enfoques: DTW, LSTM, HMM, similitud coseno", "Seleccion con justificacion tecnica (precision vs complejidad vs tiempo de entrenamiento)", "Define exactamente que features usara el modelo (angulos de huesos, cantidad por frame, normalizacion)", "Define la metrica de evaluacion (similitud 0-1, classification score, RMSE)", "Entrega ml_architecture.md antes del 17 de Abril", "El diseno ha sido revisado y aprobado por al menos otro miembro del equipo"]
    },
    {
        "summary": "[D-05] Disenar FeedbackUI.cs — layout, colores, estructura base — P5",
        "priority": "Medium", "labels": ["P5-QA"], "sprint": 1,
        "description": "Disenar el componente de UI que muestra en tiempo real que partes del cuerpo estan bien posicionadas.",
        "ac": ["Silueta/diagrama del cuerpo humano en la interfaz de Unity", "Colores por segmento: Verde >=85%, Amarillo 60-84%, Rojo <60%", "Score general del ejercicio en porcentaje visible", "Score final con desglose por segmento al terminar secuencia", "UI no tapa al robot ni interfiere con la visualizacion", "Transicion suave de colores con lerp (sin parpadeo)", "UI escalable por resolucion"]
    },

    # ── Sprint 2 ───────────────────────────────────────────────
    {
        "summary": "[C-02] Desarrollar SequenceRecorder.cs — P1",
        "priority": "High", "labels": ["P1-Unity"], "sprint": 2,
        "description": "Script C# que captura la pose completa del robot frame a frame y la guarda en formato JSON.",
        "ac": ["Bool IsRecording activable desde Inspector o codigo", "Captura todos los huesos Humanoid a 30 FPS", "Cada frame contiene: timestamp + lista de {boneName, localPosition, localRotation}", "Guarda en Application.persistentDataPath al detener grabacion", "Nombre del archivo configurable desde Inspector", "Contador de frames grabados visible en Inspector en tiempo real", "Sin degradar framerate durante la grabacion"]
    },
    {
        "summary": "[B-02] Analisis de datos Raw de Dollars Mono — P2",
        "priority": "Medium", "labels": ["P2-Python"], "sprint": 2,
        "description": "Documentar todos los mensajes OSC que envia Dollars Mono y cuales son mas utiles para el sistema de comparacion.",
        "ac": ["Documenta todos los mensajes OSC con address y tipos de valores exactos", "Identifica huesos con mayor frecuencia y mejor calidad de datos", "Identifica mensajes VMC no aprovechados actualmente", "Entrega dollars_mono_data_map.md con mapa completo", "Propone ajustes al ExternalReceiver.cs si aplica", "Incluye logs o capturas de mensajes reales como evidencia"]
    },
    {
        "summary": "[A-04] Prueba de comunicacion Python a Unity — P5",
        "priority": "High", "labels": ["P5-QA"], "sprint": 2,
        "description": "Verificar que los datos del bridge Python llegan correctamente a Unity antes de construir los modulos encima.",
        "ac": ["Mensajes OSC llegan al StudentReceiver con Unity en Play Mode", "Se verifican al menos 5 huesos distintos (posicion y rotacion en Inspector)", "DetectionConfidence se refleja correctamente", "Delay percibido < 200ms", "Reporte comunicacion_test.md con capturas del Inspector", "Problemas de firewall/puerto documentados con solucion"]
    },
    {
        "summary": "[B-03] Ajuste de filtros de suavizado (BoneFilter) — P3",
        "priority": "Medium", "labels": ["P3-3D"], "sprint": 2,
        "description": "Ajustar el parametro BoneFilter para que los movimientos del robot se vean fluidos y naturales.",
        "ac": ["BoneFilter ajustado elimina temblores visibles", "Movimientos rapidos (alzar brazo en 0.5s) sin lag excesivo", "Probado con 3 valores distintos: 0.1, 0.3, 0.7", "Documentado cual valor se eligio y por que", "Validado visualmente con video corto del robot en movimiento"]
    },
    {
        "summary": "[B-04] Validar que el robot replica movimientos correctamente — P3",
        "priority": "High", "labels": ["P3-3D"], "sprint": 2,
        "description": "Confirmar que el robot replica correctamente los movimientos del experto antes de grabar las secuencias.",
        "ac": ["Replica brazos del experto con delay < 100ms", "Replica piernas sin deformaciones", "Probado con 3 movimientos distintos (levantar brazo, agacharse, girar)", "Evidencia en video del robot replicando en tiempo real", "Problemas documentados con capturas si los hay"]
    },
    {
        "summary": "[D-06] Disenar algoritmo de puntuacion acumulada — P4",
        "priority": "High", "labels": ["P4-ML"], "sprint": 2,
        "description": "Disenar el algoritmo matematico que calcula el score total del estudiante durante y al terminar una secuencia.",
        "ac": ["Acumula score frame a frame durante la secuencia", "Score final en porcentaje (0-100%) al terminar", "Desglose por segmento corporal (cual parte mejor/peor)", "Historial de ultimos 5 intentos guardado localmente", "Mensaje final: Excelente / Bien / Sigue practicando", "Calculo reproducible: mismos datos = mismo score"]
    },

    # ── Sprint 3 ───────────────────────────────────────────────
    {
        "summary": "[C-03] Desarrollar SequencePlayer.cs — P1",
        "priority": "High", "labels": ["P1-Unity"], "sprint": 3,
        "description": "Script C# que carga un JSON de secuencia y lo reproduce en el robot con el timing correcto.",
        "ac": ["Carga JSON desde Application.persistentDataPath", "Reproduce frames respetando timestamps", "Loop y PlaybackSpeed (default 1.0) configurables", "Metodos Play(), Pause(), Stop(), Restart()", "Reinicio suave al llegar al final con Loop=true", "Seleccionable por nombre de secuencia", "Funciona aunque cambie el rig (usa nombre de hueso como clave)"]
    },
    {
        "summary": "[C-06] Panel de control en Unity (REC / PLAY / STOP / selector) — P1",
        "priority": "Medium", "labels": ["P1-Unity"], "sprint": 3,
        "description": "Panel visual en Unity para controlar grabacion y reproduccion sin modificar codigo.",
        "ac": ["Botones visibles: REC, STOP, PLAY, PAUSE", "Selector desplegable con secuencias disponibles (lee JSONs existentes)", "Indicador del estado actual: Grabando / Reproduciendo / Detenido", "Tiempo transcurrido en formato MM:SS", "Indicador rojo parpadeante en estado REC", "Compatible con las 2 secuencias de 8 tiempos", "Usable en Play Mode sin cerrar Unity"]
    },
    {
        "summary": "[C-01] Disenar formato JSON de secuencia (coordinar con P1) — P3",
        "priority": "Medium", "labels": ["P3-3D"], "sprint": 3,
        "description": "Definir el formato de archivo JSON estandar para las secuencias coordinado con P1.",
        "ac": ["Formato acordado con P1 que escribe los scripts", "Contiene metadata (nombre, fecha, FPS, duracion) y array de frames", "Cada frame tiene timestamp y bones [{name, px, py, pz, rx, ry, rz, rw}]", "Documentado en sequence_format.md", "Legible por humanos con indentacion", "JSON de ejemplo con al menos 10 frames"]
    },
    {
        "summary": "[C-04] Grabar Secuencia 1 (8 tiempos) con Dollars Mono — P3",
        "priority": "High", "labels": ["P3-3D"], "sprint": 3,
        "description": "Grabar la primera secuencia de referencia de 8 tiempos con Dollars Mono.",
        "ac": ["Exactamente 8 tiempos de duracion", "Minimo 3 repeticiones grabadas para robustez del dataset", "Grabada con Dollars Mono activo y robot replicando", "Minimo 200 frames en el JSON (>6s a 30 FPS)", "Cubre movimiento de al menos 3 partes del cuerpo", "Guardada como secuencia_01.json en la carpeta correcta", "Video de referencia grabado"]
    },
    {
        "summary": "[C-05] Grabar Secuencia 2 (8 tiempos) con Dollars Mono — P3",
        "priority": "High", "labels": ["P3-3D"], "sprint": 3,
        "description": "Grabar la segunda secuencia de referencia, notablemente diferente a la primera.",
        "ac": ["Mismos criterios que C-04", "Notablemente diferente a Secuencia 1 (diferentes grupos musculares o patron)", "Guardada como secuencia_02.json", "Tempo similar a Secuencia 1 para facilitar comparacion temporal"]
    },
    {
        "summary": "[E-inv] Implementar DTW o algoritmo de comparacion elegido — P4",
        "priority": "High", "labels": ["P4-ML"], "sprint": 3,
        "description": "Investigar e implementar el algoritmo DTW (o el elegido en E-01) para comparar secuencias de movimiento.",
        "ac": ["Algoritmo implementado segun arquitectura definida en E-01", "Tolerante a diferencias de velocidad entre secuencias", "Pruebas con datos mock documentadas", "Codigo revisado por el equipo"]
    },
    {
        "summary": "[D-03] Validacion landmarks a huesos Unity con datos mock — P5",
        "priority": "High", "labels": ["P5-QA"], "sprint": 3,
        "description": "Verificar que cada landmark 3D de MediaPipe mapea al hueso correcto en Unity, primero con datos mock.",
        "ac": ["Documento landmark_mapping.md con tabla MediaPipe landmark ID a nombre hueso Unity", "Prueba: mover brazo derecho solo hueso derecho cambia", "Prueba: mover pierna izquierda solo pierna izquierda cambia", "33 landmarks documentados (mapeados o justificada su exclusion)", "Tolerancia: visibility < 0.4 no actualiza ese hueso"]
    },

    # ── Sprint 4 ───────────────────────────────────────────────
    {
        "summary": "[D-01] Desarrollar MovementComparator.cs — calculo de angulos — P1",
        "priority": "High", "labels": ["P1-Unity"], "sprint": 4,
        "description": "Script C# que compara la pose del robot con la pose del estudiante en tiempo real usando angulos de cuaterniones.",
        "ac": ["Lee pose del robot desde Animator cada frame", "Lee pose del estudiante desde StudentReceiver", "Calcula Quaternion.Angle() por segmento corporal", "5 segmentos: BrazoIzquierdo, BrazoDerecho, PiernaIzquierda, PiernaDerecha, Torso", "Diccionario publico BodyPartScore con valores 0.0-1.0", "Metodo publico GetOverallScore() con promedio ponderado", "Solo evalua huesos con visibility > 0.4 (configurable)", "Sin NullReferenceException si faltan datos del estudiante"]
    },
    {
        "summary": "[D-02] Logica de deteccion de replica (umbrales %) — P1",
        "priority": "High", "labels": ["P1-Unity"], "sprint": 4,
        "description": "Sistema de tres niveles (verde/amarillo/rojo) para detectar en tiempo real que tan bien replica el estudiante.",
        "ac": ["Tres niveles configurables desde Inspector: Verde >=85%, Amarillo 60-84%, Rojo <60%", "Evento publico OnReplicaDetected cuando score >85% por mas de 1 segundo", "Estado actualizado cada frame", "Sistema de racha OnGoodStreak a 3 segundos consecutivos en verde", "Todos los umbrales editables sin recompilar"]
    },
    {
        "summary": "[D-04] Calibracion de proporciones corporales (pose T) — P1",
        "priority": "Medium", "labels": ["P1-Unity"], "sprint": 4,
        "description": "Sistema de calibracion al inicio de cada sesion para normalizar las proporciones del cuerpo del estudiante.",
        "ac": ["Instruccion en pantalla: Adopta la pose T brazos extendidos al frente", "Muestra de 30 frames de pose T para calcular proporciones", "Factores de escala por segmento: armScale, legScale, torsoScale", "Comparador usa factores para normalizar angulos", "Boton Recalibrar sin reiniciar Unity", "Aviso y repeticion si visibilidad insuficiente en pose T"]
    },
    {
        "summary": "[E-02] Preparar dataset de las 2 secuencias para el modelo ML — P2",
        "priority": "High", "labels": ["P2-Python"], "sprint": 4,
        "description": "Leer los JSON de las 2 secuencias y generar el dataset normalizado para entrenar el modelo.",
        "ac": ["Lee secuencia_01.json y secuencia_02.json", "Normaliza angulos de huesos (-1 a 1 o 0 a 1)", "Genera vectores de features por frame (minimo 10 angulos articulares)", "Guarda en formato compatible con modelo de P4", "Metadatos por secuencia (duracion, frames, FPS)", "Script reproducible: si los JSON cambian el dataset se regenera igual"]
    },
    {
        "summary": "[E-03] Entrenar modelo con Secuencia 1 y Secuencia 2 — P4",
        "priority": "High", "labels": ["P4-ML"], "sprint": 4,
        "description": "Entrenar el modelo ML con las 2 secuencias grabadas para comparar la pose del estudiante en tiempo real.",
        "ac": ["Entrenado con dataset de E-02", "Distingue Secuencia 1 y 2 con 80%+ accuracy en datos de prueba", "Puntuacion continua 0.0-1.0 por frame", "Modelo guardado en formato portable (.pkl, .pt, u .onnx)", "Notebook o script con graficas de resultados del entrenamiento", "Tiempo de carga del modelo < 5 segundos"]
    },
    {
        "summary": "[D-06] Implementar e integrar puntuacion al comparador — P4",
        "priority": "High", "labels": ["P4-ML"], "sprint": 4,
        "description": "Implementar e integrar el sistema de puntuacion acumulada con el MovementComparator.",
        "ac": ["Integrado con MovementComparator.cs", "Score acumulado actualizado frame a frame", "Resultado disponible para FeedbackUI", "Calculo consistente y reproducible"]
    },
    {
        "summary": "[D-03] Validacion de puntos con datos reales de MediaPipe — P5",
        "priority": "High", "labels": ["P5-QA"], "sprint": 4,
        "description": "Repetir la validacion de mapeo de landmarks con datos reales del bridge Python.",
        "ac": ["Validado con datos reales del bridge YOLO+MediaPipe corriendo", "Los 33 landmarks documentados con comportamiento real", "Sin discrepancias entre mapeo mock y mapeo real"]
    },
    {
        "summary": "[D-05] Conectar FeedbackUI.cs con datos reales del comparador — P5",
        "priority": "High", "labels": ["P5-QA"], "sprint": 4,
        "description": "Conectar el FeedbackUI con los datos reales del MovementComparator para mostrar feedback en tiempo real.",
        "ac": ["FeedbackUI lee datos de MovementComparator en tiempo real", "Colores cambian en respuesta a movimientos reales del estudiante", "Score general actualizado cada frame", "Sin lag perceptible entre movimiento y cambio de color en UI"]
    },

    # ── Sprint 5 ───────────────────────────────────────────────
    {
        "summary": "[E-04] Integrar modelo entrenado a Unity (inferencia runtime) — P2",
        "priority": "High", "labels": ["P2-Python"], "sprint": 5,
        "description": "Integrar el modelo ML entrenado al proceso Python del bridge para inferencia en tiempo real.",
        "ac": ["Modelo corre dentro del proceso Python del bridge (sin servidor adicional)", "Recibe pose del estudiante y devuelve score de similitud", "Envia /Student/MLScore (float 0.0-1.0) a Unity por OSC", "Latencia adicional < 50ms", "Funciona con las 2 secuencias, seleccionable por parametro"]
    },
    {
        "summary": "[E-05] Prueba de inferencia en tiempo real (<100ms por frame) — P2",
        "priority": "Medium", "labels": ["P2-Python"], "sprint": 5,
        "description": "Medir y verificar que el modelo ML corre en tiempo real sin lag visible.",
        "ac": ["CPU: latencia < 100ms por frame (>=10 FPS de inferencia)", "GPU: latencia < 33ms por frame (>=30 FPS)", "Reporte con metricas: FPS promedio, latencia P50/P95, uso CPU/GPU", "Sin cuelgue si persona sale del frame por mas de 2 segundos"]
    },
    {
        "summary": "[F-04] Validacion final de los 4 requerimientos minimos — P4",
        "priority": "High", "labels": ["P4-ML"], "sprint": 5,
        "description": "Validar y documentar con evidencia que el sistema cumple los 4 requerimientos minimos.",
        "ac": ["Req 1: Video demostrando deteccion de replica con feedback verde", "Req 2: Comparativa antes/despues de mejora 3D del robot", "Req 3: secuencia_01.json y secuencia_02.json existen y modelo fue entrenado con ellos", "Req 4: Documento de mapeo de 33 landmarks a huesos Unity", "Todo documentado en validacion_final.md con capturas o videos como evidencia"]
    },
    {
        "summary": "[F-01] Integracion de todos los modulos — pipeline completo — P5",
        "priority": "Highest", "labels": ["P5-QA"], "sprint": 5,
        "description": "Conectar todos los modulos del pipeline completo para que operen juntos en una sola ejecucion.",
        "ac": ["Pipeline completo operativo: Dollars Mono a robot / SequencePlayer a robot / Webcam a YOLO+MediaPipe a StudentReceiver / MovementComparator a scores / FeedbackUI en tiempo real", "Sin errores criticos en consola de Unity", "Sistema inicia en < 10 segundos desde Play", "Escena ordenada con GameObjects nombrados claramente"]
    },
    {
        "summary": "[F-02] Prueba end-to-end: experto graba a robot a estudiante evalua — P5",
        "priority": "Highest", "labels": ["P5-QA"], "sprint": 5,
        "description": "Probar el flujo completo del sistema desde grabacion hasta evaluacion del estudiante.",
        "ac": ["Prueba flujo grabacion: Dollars Mono + SequenceRecorder genera JSON valido", "Prueba flujo ensenanza: SequencePlayer reproduce correctamente", "Prueba flujo evaluacion: persona frente a camara y FeedbackUI responde", "Video de la prueba completa grabado como evidencia", "Bugs documentados en lista priorizada para F-03"]
    },

    # ── Sprint 6 ───────────────────────────────────────────────
    {
        "summary": "[F-03] Correccion de bugs Unity Core — P1",
        "priority": "High", "labels": ["P1-Unity"], "sprint": 6,
        "description": "Resolver bugs criticos y de alta prioridad encontrados en las pruebas de integracion en Unity.",
        "ac": ["Todos los bugs Critical y High de F-02 resueltos en Unity Core", "Sistema funciona 5 minutos continuos sin crashes", "Bugs Low documentados con workaround"]
    },
    {
        "summary": "[F-03] Correccion de bugs Python bridge — P2",
        "priority": "High", "labels": ["P2-Python"], "sprint": 6,
        "description": "Resolver bugs criticos del bridge Python encontrados en las pruebas.",
        "ac": ["Bugs Critical y High del bridge Python resueltos", "Parametros de deteccion ajustados para el espacio real de la demo", "Bridge estable por sesiones de 5+ minutos"]
    },
    {
        "summary": "[F-05] Assets y escena listos para demo — P3",
        "priority": "High", "labels": ["P3-3D"], "sprint": 6,
        "description": "Preparar la escena de Unity con iluminacion, camara y layout correctos para la demo final.",
        "ac": ["Iluminacion correcta en la escena", "Camara bien posicionada para la demo", "Layout y UI visible sin interferir con el robot", "Assets limpios y sin artefactos visuales"]
    },
    {
        "summary": "[F-04] Confirmar checklist y documentar resultados finales — P4",
        "priority": "High", "labels": ["P4-ML"], "sprint": 6,
        "description": "Confirmar que todos los requerimientos minimos estan cumplidos y documentar los resultados finales.",
        "ac": ["Checklist de 4 requerimientos verificado al 100%", "validacion_final.md completo con evidencias", "Documento disponible para la presentacion"]
    },
    {
        "summary": "[F-03] Correccion de bugs generales y ajuste de parametros — P5",
        "priority": "High", "labels": ["P5-QA"], "sprint": 6,
        "description": "Resolver bugs generales y ajustar parametros para la demo final.",
        "ac": ["Todos los bugs Critical y High resueltos", "Umbrales ajustados para el espacio real de la demo", "Sistema estable por 5 minutos continuos sin crashes", "Bugs Low documentados con workaround"]
    },
    {
        "summary": "[F-05] Preparacion de demo final — guion, prueba, presentacion — P5",
        "priority": "High", "labels": ["P5-QA"], "sprint": 6,
        "description": "Preparar todo para demostrar el sistema de forma clara en la presentacion final.",
        "ac": ["Guion de demo de maximo 5 minutos definido", "Escena con iluminacion correcta, camara bien posicionada, UI visible", "Demo completa probada al menos 2 veces", "Bridge arranca con un solo comando: python bridge.py", "Plan B documentado (video pregrabado si algo falla)", "Archivo de presentacion (PPT u otro) con arquitectura del sistema"]
    },
]

PRIORITY_MAP = {"Highest": "Highest", "High": "High", "Medium": "Medium", "Low": "Low"}

def build_description(description, ac_items):
    content = []
    if description:
        content.append({"type": "paragraph", "content": [{"type": "text", "text": description}]})
    if ac_items:
        content.append({"type": "paragraph", "content": [{"type": "text", "text": "Criterios de aceptacion:", "marks": [{"type": "strong"}]}]})
        items = [{"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": item}]}]} for item in ac_items]
        content.append({"type": "bulletList", "content": items})
    return {"type": "doc", "version": 1, "content": content}

def get_board_id():
    r = requests.get(f"{AGILE_URL}/board", auth=auth, headers=headers, params={"projectKeyOrId": PROJECT_KEY})
    if r.status_code != 200:
        print(f"  No se pudo obtener el board: {r.status_code}")
        return None
    values = r.json().get("values", [])
    if not values:
        print("  No se encontro ningun board.")
        return None
    board_id = values[0]["id"]
    print(f"  Board encontrado: ID={board_id} ({values[0]['name']})")
    return board_id

def create_sprints(board_id):
    sprint_ids = {}
    for num, info in SPRINTS.items():
        payload = {"name": info["name"], "startDate": f"{info['start']}T09:00:00.000Z",
                   "endDate": f"{info['end']}T23:59:00.000Z", "originBoardId": board_id}
        r = requests.post(f"{AGILE_URL}/sprint", auth=auth, headers=headers, data=json.dumps(payload))
        if r.status_code == 201:
            sid = r.json()["id"]
            sprint_ids[num] = sid
            print(f"  Sprint {num} creado: ID={sid} — {info['name']}")
        else:
            print(f"  Sprint {num} error {r.status_code}: {r.text[:120]}")
    return sprint_ids

def create_issue(issue):
    payload = {"fields": {"project": {"key": PROJECT_KEY}, "summary": issue["summary"],
                          "issuetype": {"name": "Task"}, "priority": {"name": PRIORITY_MAP.get(issue.get("priority", "Medium"), "Medium")},
                          "labels": issue.get("labels", []), "description": build_description(issue.get("description", ""), issue.get("ac", []))}}
    r = requests.post(f"{BASE_URL}/issue", auth=auth, headers=headers, data=json.dumps(payload))
    return r.status_code, r.json()

def assign_to_sprint(issue_key, sprint_id):
    r = requests.post(f"{AGILE_URL}/sprint/{sprint_id}/issue", auth=auth, headers=headers,
                      data=json.dumps({"issues": [issue_key]}))
    return r.status_code

# ── Main ────────────────────────────────────────────────────────
print("=" * 55)
print("  Dance Labo — Jira Import Script")
print("=" * 55)

print("\n[1/4] Probando conexion...")
r = requests.get(f"{BASE_URL}/project/{PROJECT_KEY}", auth=auth, headers=headers)
if r.status_code != 200:
    print(f"Error {r.status_code}: {r.text[:200]}")
    exit(1)
print(f"  Proyecto: {r.json().get('name')}")

print("\n[2/4] Buscando board Scrum...")
board_id = get_board_id()

sprint_ids = {}
if board_id:
    print("\n[3/4] Creando sprints...")
    sprint_ids = create_sprints(board_id)
else:
    print("\n[3/4] Saltando sprints (no se encontro board)")

print(f"\n[4/4] Creando {len(ISSUES)} issues...\n")
created, errors = [], []

for i, issue in enumerate(ISSUES):
    status, resp = create_issue(issue)
    if status == 201:
        key = resp.get("key", "?")
        sprint_num = issue.get("sprint")
        sprint_id  = sprint_ids.get(sprint_num)
        sprint_tag = ""
        if sprint_id:
            sc = assign_to_sprint(key, sprint_id)
            sprint_tag = f" [Sprint {sprint_num}]" if sc == 204 else f" [sprint err {sc}]"
        print(f"  [{i+1:02d}/{len(ISSUES)}] {key}{sprint_tag} — {issue['summary'][:62]}")
        created.append(key)
    else:
        err = resp.get("errors", resp.get("errorMessages", resp))
        print(f"  ERROR [{i+1:02d}]: {issue['summary'][:55]} -> {err}")
        errors.append(issue["summary"])
    time.sleep(0.4)

print(f"\n{'='*55}")
print(f"  Issues creados: {len(created)}  |  Errores: {len(errors)}")
if errors:
    print("\n  Con error:")
    for e in errors: print(f"    - {e}")
print(f"\n  https://{JIRA_DOMAIN}/jira/software/projects/{PROJECT_KEY}/boards")
print("=" * 55)