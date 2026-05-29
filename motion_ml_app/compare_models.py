"""
compare_models.py — Script de Comparación Cuantitativa
Dance Labo · Evaluación para el Reporte Final (Redes Neuronales vs Árboles de Decisión)

Este script:
1. Carga el dataset de movimientos de Dance Labo (31 secuencias, 9 clases).
2. Prepara las secuencias (con padding a 150 frames) y las aplana para el Árbol de Decisión.
3. Ejecuta 20 corridas experimentales variando hiperparámetros y divisiones de train/test.
4. Registra métricas detalladas: Pérdida (Loss/Error), Precisión (Accuracy) y Tiempos de Entrenamiento.
5. Exporta un reporte consolidado en 'data/comparacion_20_corridas.csv' listo para Excel.
"""

import os
import sys
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Asegurar encoding UTF-8 en consola Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ml.dataset_builder import MotionDataset

# Configuración básica
SEQUENCES_DIR = "data/sequences"
MAX_SEQ_LEN = 150
INPUT_SIZE = 27  # 9 huesos × 3 coords (xyz)
NUM_CLASSES = 9

# ─────────────────────────────────────────────────────────────────────────────
# 1. Definición del Clasificador LSTM (Red Neuronal)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Clasificar basándose en el último frame temporal
        out = self.fc(out[:, -1, :])
        return out

# ─────────────────────────────────────────────────────────────────────────────
# 2. Carga y Normalización de Datos
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess_data():
    dataset = MotionDataset(SEQUENCES_DIR)
    if len(dataset) == 0:
        print("[ERROR] No hay secuencias en data/sequences. Graba o copia JSONs primero.")
        sys.exit(1)
        
    X_list = []
    y_list = []
    
    for seq_tensor, label in zip(dataset.sequences, dataset.labels):
        # Padding o truncado a MAX_SEQ_LEN
        seq_len = seq_tensor.size(0)
        if seq_len < MAX_SEQ_LEN:
            padded = torch.zeros(MAX_SEQ_LEN, INPUT_SIZE)
            padded[:seq_len, :] = seq_tensor
        else:
            padded = seq_tensor[:MAX_SEQ_LEN, :]
        
        X_list.append(padded.numpy())
        y_list.append(label)
        
    X = np.array(X_list)  # (N, 150, 27)
    y = np.array(y_list)  # (N,)
    
    return X, y, dataset.label_map

# ─────────────────────────────────────────────────────────────────────────────
# 3. Función para entrenar LSTM
# ─────────────────────────────────────────────────────────────────────────────
def train_lstm(X_train, y_train, X_test, y_test, hidden_size, num_layers, epochs, lr, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear tensores
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    
    model = LSTMClassifier(hidden_size=hidden_size, num_layers=num_layers, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    final_loss = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(train_loader)
        
    training_time = time.time() - start_time
    
    # Evaluar
    model.eval()
    with torch.no_grad():
        X_te_t = X_te_t.to(device)
        outputs = model(X_te_t)
        _, predicted = torch.max(outputs, 1)
        test_acc = accuracy_score(y_test, predicted.cpu().numpy())
        
        # Train accuracy para ver sobreajuste
        outputs_tr = model(X_tr_t.to(device))
        _, predicted_tr = torch.max(outputs_tr, 1)
        train_acc = accuracy_score(y_train, predicted_tr.cpu().numpy())
        
    return final_loss, train_acc, test_acc, training_time

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline de Comparación: 20 Corridas
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison():
    print("=" * 65)
    print("  Dance Labo — Generador de 20 Corridas Comparativas para Reporte")
    print("=" * 65)
    
    X, y, label_map = load_and_preprocess_data()
    print(f"\nDataset cargado exitosamente:")
    print(f"  - Muestras totales: {X.shape[0]}")
    print(f"  - Dimensiones: {X.shape[1]} frames × {X.shape[2]} características")
    print(f"  - Clases: {list(label_map.keys())}\n")
    
    # Lista de combinaciones de hiperparámetros para las 20 corridas
    # Varía la división de train/test, tamaño oculto LSTM, épocas y profundidad de Árbol de Decisión
    configs = []
    for i in range(1, 21):
        test_size = 0.20 if i % 2 == 0 else 0.30
        
        # Parámetros LSTM
        hidden_size = 32 if i <= 10 else 64
        num_layers = 1 if i % 3 == 0 else 2
        epochs = 60 + (i * 4)  # entre 64 y 140 épocas
        lr = 0.005 if i % 4 == 0 else 0.001
        batch_size = 4
        
        # Parámetros Árbol
        criterion = "gini" if i % 2 == 0 else "entropy"
        max_depth = None if i > 15 else (3 + (i % 6))
        min_samples_split = 2 if i % 3 == 0 else 3
        
        configs.append({
            "run_id": i,
            "test_size": test_size,
            "lstm": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size
            },
            "tree": {
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            }
        })
        
    results = []
    
    for c in configs:
        run_id = c["run_id"]
        ts = c["test_size"]
        
        # División train/test
        # Para asegurar que la división varíe pero sea estable por corrida
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, random_state=42 + run_id
        )
        
        # Aplanar datos para el Árbol de Decisión (N, 150*27 = 4050 features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # --- 1. Entrenar Árbol de Decisión ---
        clf_tree = DecisionTreeClassifier(
            criterion=c["tree"]["criterion"],
            max_depth=c["tree"]["max_depth"],
            min_samples_split=c["tree"]["min_samples_split"],
            random_state=42 + run_id
        )
        
        start_tree = time.time()
        clf_tree.fit(X_train_flat, y_train)
        time_tree = time.time() - start_tree
        
        y_pred_tree_tr = clf_tree.predict(X_train_flat)
        y_pred_tree_te = clf_tree.predict(X_test_flat)
        
        tree_train_acc = accuracy_score(y_train, y_pred_tree_tr)
        tree_test_acc = accuracy_score(y_test, y_pred_tree_te)
        # Error = 1 - Accuracy
        tree_train_err = 1.0 - tree_train_acc
        tree_test_err = 1.0 - tree_test_acc
        
        # --- 2. Entrenar LSTM (Red Neuronal) ---
        lstm_loss, lstm_tr_acc, lstm_te_acc, time_lstm = train_lstm(
            X_train, y_train, X_test, y_test,
            hidden_size=c["lstm"]["hidden_size"],
            num_layers=c["lstm"]["num_layers"],
            epochs=c["lstm"]["epochs"],
            lr=c["lstm"]["lr"],
            batch_size=c["lstm"]["batch_size"]
        )
        lstm_train_err = 1.0 - lstm_tr_acc
        lstm_test_err = 1.0 - lstm_te_acc
        
        print(f"Corrida {run_id:02d}/20 | LSTM Test Acc: {lstm_te_acc:.2%} | Tree Test Acc: {tree_test_acc:.2%}")
        
        results.append({
            "Corrida": run_id,
            "Split Train/Test": f"{int((1-ts)*100)}/{int(ts*100)}",
            "LSTM Hidden Size": c["lstm"]["hidden_size"],
            "LSTM Capas": c["lstm"]["num_layers"],
            "LSTM Épocas": c["lstm"]["epochs"],
            "LSTM LR": c["lstm"]["lr"],
            "LSTM Train Loss": round(lstm_loss, 5),
            "LSTM Train Error": round(lstm_train_err, 4),
            "LSTM Test Error": round(lstm_test_err, 4),
            "LSTM Accuracy Test": round(lstm_te_acc, 4),
            "LSTM Tiempo (s)": round(time_lstm, 4),
            "Árbol Criterio": c["tree"]["criterion"],
            "Árbol Max Depth": c["tree"]["max_depth"] if c["tree"]["max_depth"] else "None",
            "Árbol Min Split": c["tree"]["min_samples_split"],
            "Árbol Train Error": round(tree_train_err, 4),
            "Árbol Test Error": round(tree_test_err, 4),
            "Árbol Accuracy Test": round(tree_test_acc, 4),
            "Árbol Tiempo (s)": round(time_tree, 4),
        })

    # Guardar resultados en CSV
    csv_path = "data/comparacion_20_corridas.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n\u2705 Reporte guardado con éxito en '{csv_path}'")
    
    # Imprimir resumen de promedios para copiar rápido
    avg_lstm_acc = np.mean([r["LSTM Accuracy Test"] for r in results])
    avg_tree_acc = np.mean([r["Árbol Accuracy Test"] for r in results])
    avg_lstm_time = np.mean([r["LSTM Tiempo (s)"] for r in results])
    avg_tree_time = np.mean([r["Árbol Tiempo (s)"] for r in results])
    
    print("\n" + "=" * 50)
    print("               RESUMEN DE METRICAS")
    print("=" * 50)
    print(f"Promedio Accuracy LSTM (Red Neuronal): {avg_lstm_acc:.2%}")
    print(f"Promedio Accuracy Árbol de Decisión:  {avg_tree_acc:.2%}")
    print(f"Tiempo promedio ent. LSTM:            {avg_lstm_time:.4f} s")
    print(f"Tiempo promedio ent. Árbol:           {avg_tree_time:.4f} s")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_comparison()
