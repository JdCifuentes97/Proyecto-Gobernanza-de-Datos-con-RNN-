import pandas as pd
import re

# Ruta al archivo CSV original
input_file_path = r'Ruta del archivo'

# Patrón específico que estamos buscando
pattern = re.compile(r'910;SEAS BOGOTA ROJO ASOCIADOS LTDA5;PATRIMONIALES')

# Función para identificar si una fila contiene el patrón específico
def has_specific_format_error(row):
    # Convierte la fila en una cadena
    row_str = ';'.join(row.astype(str))
    # Busca el patrón específico en la fila
    return bool(pattern.search(row_str))

# Lee el archivo CSV en un DataFrame
df = pd.read_csv(input_file_path, delimiter=';', header=None)

# Encuentra las filas con el patrón específico
matching_rows = df[df.apply(has_specific_format_error, axis=1)]

# Elimina las filas con el patrón específico
df_cleaned = df[~df.apply(has_specific_format_error, axis=1)]

# Guarda el DataFrame limpio en el mismo archivo CSV
df_cleaned.to_csv(input_file_path, index=False, header=False, sep=';')

# Imprime las filas eliminadas
print("Filas eliminadas:")
print(matching_rows.to_string(index=False, header=False))

# Imprime el número de filas eliminadas
print(f'\nNúmero de filas con el patrón específico eliminadas: {len(matching_rows)}')
