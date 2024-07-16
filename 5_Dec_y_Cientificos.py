import pandas as pd

# Ruta del archivo CSV
file_path = "Ruta del archivo CSV"

# Leer el archivo CSV
df = pd.read_csv(file_path, delimiter=';')

# Función para convertir valores a enteros
def convert_to_int(value):
    try:
        return int(float(value.replace(',', '')))
    except ValueError:
        return value  # Mantener el valor original si la conversión falla

# Lista para almacenar ejemplos de cambios
examples = []

# Convertir los valores de 'imp prima' a enteros, manejando notación científica y eliminando comas
def conversion_and_logging(value):
    original_value = str(value)
    converted_value = convert_to_int(original_value)
    if 'E+' in original_value or ',' in original_value:  # Verificar si el valor estaba en notación científica o tenía comas
        examples.append((original_value, converted_value))
    return converted_value

df['imp prima'] = df['imp prima'].apply(conversion_and_logging)

# Guardar el DataFrame de nuevo en el archivo CSV
df.to_csv(file_path, index=False, sep=';')

# Imprimir 10 ejemplos de cambios realizados
for original, converted in examples[:10]:
    print(f"Original: {original}, Convertido: {converted}")

print("Decimales eliminados y archivo CSV actualizado.")
