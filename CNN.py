# Importar las librerías necesarias
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("Dispositivos disponibles:", tf.config.list_physical_devices())
# Leer el dataset (se asume que el dataset ya está preprocesado y listo para usar)
try:
    aqi_data = pd.read_csv(r"C:\Users\andre\OneDrive\Documentos\TF-Paralela\US_AQI.csv", index_col=0)
    
    print("Sample of AQI Dataset:")
    print(aqi_data.head())
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo CSV en la ruta especificada.")
except pd.errors.EmptyDataError:
    print("Error: El archivo CSV está vacío.")
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")

# Crear un modelo CNN simple
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Mostrar el resumen del modelo
model.summary()