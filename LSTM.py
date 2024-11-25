# Importar las librerías necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Leer el dataset (se asume que el dataset ya está preprocesado y listo para usar)
try:
    aqi_data = pd.read_csv(r"C:\Users\andre\OneDrive\Documentos\TF-Paralela\US_AQI.csv", index_col=0)
    aqi_data = aqi_data.head(3_000_000)
    print("Sample of AQI Dataset:")
    print(aqi_data.head())
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo CSV en la ruta especificada.")
except pd.errors.EmptyDataError:
    print("Error: El archivo CSV está vacío.")
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")

# 2. Normalización de los datos
# Ejemplo de cómo podrías normalizar los datos (asumiendo que 'train_windows', 'test_windows', etc. ya existen)
aqi_mean = np.array(train_windows).mean()
aqi_std = np.array(train_windows).std()
train_windows = (np.array(train_windows) - aqi_mean) / aqi_std
test_windows = (np.array(test_windows) - aqi_mean) / aqi_std
val_windows = (np.array(val_windows) - aqi_mean) / aqi_std

# 3. División en X e y
train_X = train_windows[:, :-7].reshape(train_windows.shape[0], 30, 1)
train_y = train_windows[:, -7:]
test_X = test_windows[:, :-7].reshape(test_windows.shape[0], 30, 1)
test_y = test_windows[:, -7:]
val_X = val_windows[:, :-7].reshape(val_windows.shape[0], 30, 1)
val_y = val_windows[:, -7:]

# Muestra de datos preprocesados
print("SAMPLE OF PREPROCCESED DATA SET")
print("======================================")
print("X:", train_X[0])
print("y:", train_y[0])

# Definir la arquitectura del modelo
model = Sequential([
    Bidirectional(LSTM(150, dropout=0.3, return_sequences=True, input_shape=(30, 1))),
    Bidirectional(LSTM(150, dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(150, dropout=0.3)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(7)
])

# Compilar el modelo
model.compile(
    loss='mae',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['mse']
)

# Configurar EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25
)

# Mostrar el resumen del modelo
print("MODEL SUMMARY")
print("==========================")
model.summary()
