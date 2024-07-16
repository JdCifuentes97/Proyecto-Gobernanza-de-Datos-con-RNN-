import pandas as pd
import re

# Ruta del archivo CSV
file_path = "Ruta del archivo CSV"

# Cargar los datos
data = pd.read_csv(file_path, delimiter=';', low_memory=False)


# Función para corregir los formatos de fecha que tienen hora
def corregir_fechas_con_hora(fecha):

    # Eliminar cualquier cosa después del año (incluyendo espacios y horas)
    fecha = re.sub(r'(\d{1,2}/\d{1,2}/\d{4}).*', r'\1', fecha)

    # Convertir de mm/dd/yyyy a dd/mm/yyyy
    try:
        fecha_dt = pd.to_datetime(fecha, format='%m/%d/%Y', errors='coerce')
        if fecha_dt is not pd.NaT:
            return fecha_dt.strftime('%d/%m/%Y')
    except Exception as e:
        print(f"Error al convertir la fecha: {fecha}. Error: {e}")

    return fecha


# Aplicar la corrección de fechas en toda la columna
data['fec endoso'] = data['fec endoso'].apply(corregir_fechas_con_hora)

# Guardar los cambios en el mismo archivo CSV
data.to_csv(file_path, sep=';', index=False)

print("Fechas corregidas y guardadas en el archivo.")
