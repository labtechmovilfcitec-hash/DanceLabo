"""
train.py — Script de Entrenamiento del Modelo LSTM
Dance Labo · E-03 / E-01

Uso:
    cd motion_ml_app
    python train.py

Requisitos previos:
    - Tener secuencia_01.json y secuencia_02.json en data/sequences/
    - pip install torch (ya en requirements.txt)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.model import MotionLSTMGenerator
from ml.dataset_builder import MotionDataset

# ── Configuración ─────────────────────────────────────────────────────────────
DATA_DIR     = "../data/sequences"   # relativo a motion_ml_app/
MODEL_PATH   = "../data/motion_model.pt"
LABEL_PATH   = "../data/label_map.pt"
EPOCHS       = 150
BATCH_SIZE   = 4
LR           = 1e-3
HIDDEN_SIZE  = 256
NUM_LAYERS   = 3
OUTPUT_SIZE  = 27   # 9 huesos Mixamo × 3 coords (xyz)



os.makedirs("../data", exist_ok=True)



def collate_fn(batch):
    """Padding para que todos los tensores del batch tengan el mismo tamaño."""
    sequences, labels = zip(*batch)
    max_len = max(s.size(0) for s in sequences)
    padded = torch.zeros(len(sequences), max_len, sequences[0].size(1))
    for i, s in enumerate(sequences):
        padded[i, :s.size(0)] = s
    return padded, torch.tensor(labels)


def train():
    print("=" * 50)
    print("  Dance Labo — Entrenamiento LSTM")
    print("=" * 50)

    # 1. Cargar dataset
    dataset = MotionDataset(DATA_DIR)

    if len(dataset) == 0:
        print(f"\n[ERROR] No se encontraron secuencias en '{DATA_DIR}'")
        print("  Graba primero secuencia_01.json y secuencia_02.json")
        return

    print(f"\nMovimientos encontrados: {dataset.label_map}")
    print(f"Total de muestras: {len(dataset)}")
    print(f"Longitud máxima de secuencia: {dataset.max_length} frames")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, drop_last=False)

    # 2. Crear modelo
    num_classes = len(dataset.label_map)
    model = MotionLSTMGenerator(
        num_classes=num_classes,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,      # 27 (9 huesos Mixamo × xyz)
        max_seq_length=dataset.max_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\nDispositivo: {device}")
    print(f"Clases: {num_classes}  |  Epochs: {EPOCHS}  |  LR: {LR}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 3. Entrenamiento
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for seqs, labels in loader:
            seqs   = seqs.to(device)      # (B, T, 27)
            labels = labels.to(device)    # (B,)

            seq_len = seqs.size(1)
            generated = model(labels, seq_length=seq_len)  # (B, T, 27)

            loss = criterion(generated, seqs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        # Guardar el mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(dataset.label_map, LABEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            bar = "█" * int((1 - min(avg_loss, 1)) * 20)
            print(f"Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} |{bar}")

    print(f"\n✅ Entrenamiento completo.")
    print(f"   Modelo guardado en: {MODEL_PATH}")
    print(f"   Label map en:       {LABEL_PATH}")
    print(f"   Loss final:         {best_loss:.6f}")


if __name__ == "__main__":
    train()
