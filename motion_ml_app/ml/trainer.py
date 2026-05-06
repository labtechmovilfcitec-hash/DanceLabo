import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.dataset_builder import MotionDataset
from ml.model import MotionLSTMGenerator

def train_model(epochs=150, batch_size=4, lr=0.001):
    print("Iniciando entrenamiento...")
    
    dataset = MotionDataset()
    if len(dataset) == 0:
        print("Error: El dataset esta vacio. Graba algunas secuencias en la UI primero.")
        return
        
    print(f"Dataset cargado con {len(dataset)} secuencias.")
    print(f"Clases detectadas: {dataset.label_map}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Configuracion del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Inicializar modelo
    num_classes = len(dataset.label_map)
    # output_size = 27 (9 vectores * 3 coordenadas)
    model = MotionLSTMGenerator(num_classes=num_classes, max_seq_length=dataset.max_length).to(device)
    
    # Loss y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Bucle de entrenamiento
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass: El modelo intenta generar la secuencia dado el ID de la etiqueta
            optimizer.zero_grad()
            
            # generated_seq shape: (batch_size, max_seq_length, 27)
            generated_seq = model(labels, seq_length=sequences.size(1))
            
            # Calcular la diferencia (error) entre lo generado y el movimiento real
            loss = criterion(generated_seq, sequences)
            
            # Backward pass y optimizacion
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    # Guardar el modelo entrenado
    os.makedirs('data/models', exist_ok=True)
    
    save_info = {
        'model_state_dict': model.state_dict(),
        'label_map': dataset.label_map,
        'max_seq_length': dataset.max_length
    }
    
    torch.save(save_info, 'data/models/motion_lstm.pt')
    print("Entrenamiento completado. Modelo guardado en 'data/models/motion_lstm.pt'")

if __name__ == "__main__":
    train_model()
