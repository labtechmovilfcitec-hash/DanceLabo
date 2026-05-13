"""
ml/trainer.py  —  Entrenador LSTM (Dance Labo)
Equivalente completo a train.py pero importable desde la UI.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset_builder import MotionDataset
from ml.model import MotionLSTMGenerator

# ── Rutas (relativas a motion_ml_app/) ────────────────────────────────────────
MODEL_PATH  = "data/motion_model.pt"
LABEL_PATH  = "data/label_map.pt"
DATA_DIR    = "data/sequences"

# ── Hiperparámetros ────────────────────────────────────────────────────────────
EPOCHS      = 150
BATCH_SIZE  = 4
LR          = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS  = 3
OUTPUT_SIZE = 27   # 9 huesos Mixamo × 3 coords (xyz)


def _collate_fn(batch):
    """Padding dinámico para que todos los tensores del batch tengan el mismo largo."""
    sequences, labels = zip(*batch)
    max_len = max(s.size(0) for s in sequences)
    padded = torch.zeros(len(sequences), max_len, sequences[0].size(1))
    for i, s in enumerate(sequences):
        padded[i, :s.size(0)] = s
    return padded, torch.tensor(labels)


def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    """
    Entrena el modelo LSTM y guarda:
      - data/motion_model.pt   (pesos del mejor epoch)
      - data/label_map.pt      (dict nombre → índice)
    """
    print("=" * 50)
    print("  Dance Labo — Entrenamiento LSTM")
    print("=" * 50)

    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Cargar dataset
    dataset = MotionDataset(DATA_DIR)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No se encontraron secuencias en '{DATA_DIR}'. "
            "Graba al menos una secuencia antes de entrenar."
        )

    print(f"\nMovimientos encontrados: {dataset.label_map}")
    print(f"Total de muestras:       {len(dataset)}")
    print(f"Longitud máx. secuencia: {dataset.max_length} frames")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    # 2. Crear modelo
    num_classes = len(dataset.label_map)
    model = MotionLSTMGenerator(
        num_classes=num_classes,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        max_seq_length=dataset.max_length,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\nDispositivo: {device}")
    print(f"Clases: {num_classes}  |  Epochs: {epochs}  |  LR: {lr}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 3. Bucle de entrenamiento
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for seqs, labels in loader:
            seqs   = seqs.to(device)
            labels = labels.to(device)

            seq_len   = seqs.size(1)
            generated = model(labels, seq_length=seq_len)

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
            os.makedirs("data", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(dataset.label_map, LABEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            bar = "#" * int((1 - min(avg_loss, 1)) * 20)
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} |{bar}")

    print(f"\n✅ Entrenamiento completo.")
    print(f"   Modelo guardado en: {MODEL_PATH}")
    print(f"   Label map en:       {LABEL_PATH}")
    print(f"   Loss final:         {best_loss:.6f}")
