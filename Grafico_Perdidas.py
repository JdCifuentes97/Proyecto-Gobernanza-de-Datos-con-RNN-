import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Cargar el archivo CSV
file_path = 'Ruta Archivo CSV'
df = pd.read_csv(file_path, delimiter=';')

# Convertir la columna de fechas a tipo datetime sin formato específico
df['fec endoso'] = pd.to_datetime(df['fec endoso'], errors='coerce')

# Filtrar los datos para solo incluir los registros con el error en expedición
df_error = df[df['tipo endo'] == 'ERROR EN EXPEDICION'].copy()

# Verificar si el DataFrame filtrado tiene datos
if df_error.empty:
    print("No hay registros con 'ERROR EN EXPEDICION' en 'tipo endo'.")
else:
    # Agrupar por año y calcular la cantidad de errores
    df_error['Año'] = df_error['fec endoso'].dt.year

    # Excluir el año 2023
    df_error = df_error[df_error['Año'] != 2023]

    errors_by_year = df_error.groupby('Año').size()

    # Verificar si la agrupación tiene datos
    if errors_by_year.empty:
        print("No hay datos de errores por año.")
    else:
        # Preparación de datos para el modelo LSTM
        df_error['AñoMes'] = df_error['fec endoso'].dt.to_period('M')
        errors_by_month = df_error.groupby('AñoMes').size().to_frame(name='Cantidad_Errores')
        errors_by_month.index = errors_by_month.index.to_timestamp()

        # Normalizar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(errors_by_month)

        # Crear datos para el modelo LSTM
        def create_dataset(dataset, time_step=1):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 12  # Usaremos 12 meses anteriores para predecir el siguiente mes
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Dividir los datos en conjunto de entrenamiento y prueba
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Construcción del modelo LSTM con ajuste en Dropout
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.3))  # Aumentar Dropout a 0.3
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.3))  # Aumentar Dropout a 0.3
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Implementar parada temprana
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Entrenamiento del modelo
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop])

        # Graficar la curva de pérdida (Loss Curve)
        plt.figure()
        plt.plot(history.history['loss'], label='Loss del entrenamiento')
        plt.plot(history.history['val_loss'], label='Loss de la validación')
        plt.title('Curva de Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()
