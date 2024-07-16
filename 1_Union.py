import pandas as pd
import os
from pathlib import Path
import chardet

# Función para detectar la codificación de un archivo
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Directorio donde se encuentran los archivos CSV
csv_folder = Path(r'ruta del archivo')

# Lista para almacenar los DataFrames
dataframes = []

# Iterar sobre los archivos en el directorio
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        file_path = csv_folder / filename
        try:
            # Detectar la codificación del archivo
            encoding = detect_encoding(file_path)
            # Leer el archivo CSV con la codificación detectada
            df = pd.read_csv(file_path, delimiter=';', encoding=encoding)
            dataframes.append(df)
        except UnicodeDecodeError as e:
            print(f"Error de codificación en el archivo {file_path}: {e}")

# Concatenar todos los DataFrames en uno solo
combined_df = pd.concat(dataframes, ignore_index=True)

# Rellenar las celdas vacías en la columna 'sn enlace' con 0 si existe
if 'sn enlace' in combined_df.columns:
    combined_df['sn enlace'] = combined_df['sn enlace'].fillna(0)

# Eliminar las columnas 'sn enlace' y 'nro pol' si existen
columns_to_drop = ['sn enlace', 'nro pol']
combined_df = combined_df.drop(columns=[col for col in columns_to_drop if col in combined_df.columns])

# Guardar el DataFrame combinado en un nuevo archivo CSV
combined_df.to_csv(csv_folder / 'resultado_combinado.csv', index=False, sep=';')#Nombre de como se llamara el archivo combinado

print("Archivos combinados, modificados y guardados exitosamente.")
