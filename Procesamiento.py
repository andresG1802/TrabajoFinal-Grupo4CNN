import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta mensajes de advertencia de TensorFlow

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Carga el conjunto de datos
aqi_data = pd.read_csv(r"C:\Users\andre\OneDrive\Documentos\TF-Paralela\Procesamiento100Mil.csv", index_col=0)

# Ignorar la columna de fecha y asegurar que solo se usen columnas numéricas
aqi_data = aqi_data.apply(pd.to_numeric, errors='coerce').dropna()

# Escalar los datos
scaler = MinMaxScaler()
aqi_data_scaled = scaler.fit_transform(aqi_data)

# Parámetros del modelo
timesteps = 30  # Número de pasos de tiempo de entrada
features = aqi_data.shape[1]  # Número de características (columnas)
future_steps = 7  # Número de pasos a predecir

# Función para crear secuencias de entrada y salida
def create_sequences(data, timesteps, future_steps):
    X, y = [], []
    for i in range(len(data) - timesteps - future_steps):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps:i + timesteps + future_steps, 0])  # Suponiendo que la salida es de la primera columna
    return np.array(X), np.array(y)

# Generar secuencias
X, y = create_sequences(aqi_data_scaled, timesteps, future_steps)

# Dividir en entrenamiento y validación
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar las dimensiones de las entradas
print("train_X shape:", train_X.shape)  # Debería ser (muestras, 30, características)
print("train_y shape:", train_y.shape)  # Debería ser (muestras, 7)

# Definir el modelo
model = Sequential([
    Input(shape=(timesteps, features)),
    Bidirectional(LSTM(150, dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(150, dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(150, dropout=0.3)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(future_steps)  # Número de pasos de salida
])

# Compilar el modelo
model.compile(
    loss='mae',
    optimizer=Adam(learning_rate=1e-4),  # Ajustado a un valor más común
    metrics=['mse']
)

# Definir EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Reducido para entrenamientos más cortos
    restore_best_weights=True
)

# Entrenar el modelo
history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# Imprimir resumen del modelo
model.summary()
