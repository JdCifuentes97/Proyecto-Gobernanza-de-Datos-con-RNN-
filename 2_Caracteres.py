import pandas as pd
from pathlib import Path
import re

# Ruta del archivo combinado
combined_file_path = Path(r'Ruta del archivo')

def process_csv(file_path):
    # Cargar el archivo combinado
    df = pd.read_csv(file_path, delimiter=';')

    # Definir una función para reemplazar los caracteres problemáticos usando regex
    def replace_problematic_strings(text):
        # Expresiones regulares para capturar variantes de 'BOGOTÁ'
        replacements = {
            r'Á': 'A', r'É': 'E', r'Í': 'I', r'Ó': 'O', r'Ú': 'U',
            r'': 'E', r'ÃI': 'I', r'ÃO': 'O', r'ÃA': 'A',
            r'': 'O', r'': 'I', r'': 'A',
            r'ÃI': 'I', r'ÃO': 'O', r'ÃA': 'A',
            r'BOGOTÃ': 'BOGOTA', r'BOGOTÃ': 'BOGOTA', r'BOGOTAA': 'BOGOTA',
            r'IBAGUÃ': 'IBAGUE', r'IBAGUÃ': 'IBAGUE', r'IBAGUEE': 'IBAGUE',
            r'MEDELLÃN': 'MEDELLIN', r'MEDELLIN': 'MEDELLIN',
            r'MONTERÃA': 'MONTERIA', r'MONTERIA': 'MONTERIA',
            r'APARTAD': 'APARTAD', r'APARTADO': 'APARTADO', r'APARTADÃO': 'APARTADO',
            r'CIÓN': 'ION', r'ION': 'CION', r'GESTIÃON': 'GESTION',
            r'FÃCIL': 'FACIL', r'FACIL': 'FACIL',
            r'IÃN': 'ION', r'TION': 'TION',r'Ã‘A‘': 'NA',r'Ã‘O‘': 'NO',r'â€“‘': '&'
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text

    # Aplicar la función de reemplazo en todo el DataFrame
    df = df.applymap(lambda x: replace_problematic_strings(str(x)) if isinstance(x, str) else x)

    # Guardar el DataFrame modificado en un nuevo archivo CSV
    df.to_csv(file_path, index=False, sep=';')

    print("Texto reemplazado exitosamente en todo el archivo.")

# Ejecutar la función dos veces para evitar errores
process_csv(combined_file_path)
process_csv(combined_file_path)