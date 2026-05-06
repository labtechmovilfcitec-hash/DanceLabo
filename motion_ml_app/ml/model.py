import torch
import torch.nn as nn

class MotionLSTMGenerator(nn.Module):
    """
    Modelo generativo usando LSTM.
    Recibe el ID o nombre codificado del movimiento que debe ejecutar, 
    y genera (predice) la secuencia temporal completa de posiciones corporales.
    """
    def __init__(self, num_classes, hidden_size=256, num_layers=3, output_size=99, max_seq_length=150):
        """
        num_classes: Cuantos movimientos diferentes conoce
        hidden_size: Memoria del LSTM
        num_layers: Profundidad del LSTM
        output_size: 99 (33 landmarks * 3 coordenadas x,y,z)
        max_seq_length: Longitud maxima de cuadros a generar
        """
        super(MotionLSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Mapea el ID de clase (ej: "Macarena" = 1) a un vector denso (espacio latente)
        self.embedding = nn.Embedding(num_classes, hidden_size)
        
        # La red LSTM se encarga de extender el conocimiento a lo largo del tiempo
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Convierte el estado oculto del LSTM de vuelta a coordenadas 3D (output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_length=None):
        # x shape: (batch_size,) - contiene los IDs de los movimientos a generar
        batch_size = x.size(0)
        
        # 1. Obtener la "firma" o embedding del movimiento
        # embedded shape: (batch_size, hidden_size)
        embedded = self.embedding(x)
        
        # 2. Copiamos la firma a lo largo del tiempo (como "semilla" para cada cuadro)
        seq_len = seq_length if seq_length is not None else self.max_seq_length
        # repeated shape: (batch_size, seq_len, hidden_size)
        repeated_embedded = embedded.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 3. Inicializamos la memoria del LSTM en cero
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # 4. Procesar a traves del tiempo
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(repeated_embedded, (h0, c0))
        
        # 5. Mapear cada paso de memoria a las posiciones fisicas (landmarks)
        # out shape: (batch_size, seq_len, 99)
        generated_sequence = self.fc(out) 
        
        return generated_sequence
