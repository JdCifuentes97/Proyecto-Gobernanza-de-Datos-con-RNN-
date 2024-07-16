import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Cargar el archivo CSV
file_path = 'Ruta Archivo CSV'
df = pd.read_csv(file_path, delimiter=';')

# Convertir la columna de fechas a tipo datetime especificando el formato
df['fec endoso'] = pd.to_datetime(df['fec endoso'], format='%d/%m/%Y', errors='coerce')

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
        # Visualizar la cantidad de errores agrupados por año
        plt.figure(figsize=(10, 6))
        bars = errors_by_year.plot(kind='bar')
        plt.xlabel('Año')
        plt.ylabel('Cantidad de Errores')
        plt.title('Cantidad de ERRORES EN EXPEDICION por Año')
        plt.xticks(rotation=45)

        # Añadir etiquetas con la cantidad de errores dentro de cada columna
        for bar in bars.patches:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom')

        plt.show()

        # Preparación de datos para el modelo LSTM
        # Crear un DataFrame para la serie temporal de errores por mes
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

        # Construcción del modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenamiento del modelo
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        # Predecir los errores para cada mes de 2024
        future_steps = 12  # Queremos predecir los próximos 12 meses
        last_data = scaled_data[-time_step:]
        future_predictions = []

        for _ in range(future_steps):
            pred = model.predict(last_data.reshape(1, time_step, 1))
            future_predictions.append(pred[0][0])
            last_data = np.append(last_data[1:], pred[0][0]).reshape(time_step, 1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Crear un DataFrame con las predicciones ajustadas
        future_dates = pd.date_range(start='2024-01-01', periods=future_steps, freq='M')
        future_errors = pd.DataFrame(future_predictions, index=future_dates, columns=['Cantidad_Errores'])

        # Meses en español
        months_es = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre',
                     'Noviembre', 'Diciembre']

        # Graficar las predicciones ajustadas para 2024
        plt.figure(figsize=(12, 6))
        future_errors.index = future_errors.index.month - 1  # Usar índices 0-11 para los meses
        future_errors.index = [months_es[i] for i in future_errors.index]  # Asignar nombres de meses en español
        bars = future_errors.plot(kind='bar', legend=False)

        # Añadir etiquetas con la cantidad de errores dentro de cada columna
        for bar in bars.patches:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom')

        plt.ylim(0, 450)  # Ajustar el límite superior del eje Y
        plt.xlabel('Mes')
        plt.ylabel('Cantidad de Errores')
        plt.title('Predicción de Cantidad de ERRORES EN EXPEDICION para 2024')
        plt.xticks(rotation=45)
        plt.show()

        print(future_errors)

        # Crear un DataFrame con los conteos por año y agregar las predicciones de 2024
        historical_errors = errors_by_year.copy()
        historical_errors[2024] = future_predictions.sum()

        # Calcular el valor de 'imp prima' por año
        imp_prima_by_year = df_error.groupby('Año')['imp prima'].sum()

        # Obtener el promedio de 'imp prima' por error en los años 2020 a 2022
        errors_2020_2022 = df_error[df_error['Año'].isin([2020, 2021, 2022])]
        average_imp_prima_per_error = errors_2020_2022['imp prima'].sum() / errors_2020_2022.shape[0] if not errors_2020_2022.empty else 0

        # Calcular el valor probable para el año 2024
        probable_imp_prima_2024 = future_predictions.sum() * average_imp_prima_per_error

        # Crear un DataFrame con los valores de 'imp prima' por año y agregar la predicción para 2024
        historical_imp_prima = imp_prima_by_year.copy()
        historical_imp_prima[2024] = probable_imp_prima_2024

        # Graficar la cantidad de errores por año y mostrar el valor de 'imp prima' encima de cada barra
        plt.figure(figsize=(12, 6))
        bars = plt.bar(historical_errors.index, historical_errors,
                       color=['blue' if year != 2024 else 'orange' for year in historical_errors.index])

        # Añadir etiquetas con el valor de 'imp prima' dentro de cada columna
        for bar, value in zip(bars, historical_imp_prima):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{value:,.0f}', ha='center', va='bottom')

        plt.xlabel('Año')
        plt.ylabel('Cantidad de Errores')
        plt.title('Conteo de Errores por Año y Predicción de Errores para 2024')
        plt.xticks(rotation=45)
        plt.show()

        print(historical_errors)

        # Crear un DataFrame con los resultados para guardar en CSV
        results = pd.DataFrame({
            'Año': list(historical_errors.index),
            'Cantidad de Errores': list(historical_errors.values),
            'Valor Perdidas (imp prima)': list(historical_imp_prima.values)
        })

        # Guardar el DataFrame en un archivo CSV
        results.to_csv('E:\\JULIAN\\Seminario\\CSV\\resultados_imp_prima.csv', index=False)

        print(f"Valor de 'imp prima' por año:")
        print(historical_imp_prima)
        print(f"Valor probable de 'imp prima' para el año 2024 basado en la cantidad de errores predicha:")
        print(probable_imp_prima_2024)
