"""
generate_chart.py — Script para generar la gráfica comparativa de Accuracy y Guardarla en Disco
"""
import os
import sys

# Forzar encoding UTF-8 en consola
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Instalando matplotlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    import numpy as np

# Configurar estética profesional
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(7, 5.5))

# Datos
models = ['Árbol de Decisión\n(Baseline)', 'Red Neuronal LSTM\n(Propuesto)']
accuracy = [65.43, 75.14]
colors = ['#d9534f', '#5cb85c'] # Rojo suave y verde profesional

# Dibujar barras
bars = ax.bar(models, accuracy, color=colors, width=0.5, edgecolor='#2c3e50', linewidth=1.2, alpha=0.9)

# Ajustes de ejes y títulos
ax.set_ylabel('Exactitud promedio (Accuracy %)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Comparativa de Rendimiento en Clasificación de Baile\n(Promedio de 20 Corridas Experimentales)', 
             fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
ax.set_ylim(0, 100)

# Agregar etiquetas de valor arriba de las barras
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 puntos de offset vertical
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2c3e50')

# Mejorar cuadrícula y bordes
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#bdc3c7')
ax.spines['bottom'].set_color('#bdc3c7')

plt.tight_layout()

# Guardar la gráfica en data/
output_path = os.path.join("data", "grafica_accuracy.png")
os.makedirs("data", exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()

print(f"¡Gráfica comparativa generada exitosamente y guardada en '{output_path}'!")
