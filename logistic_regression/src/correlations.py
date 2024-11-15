import pandas as pd
import numpy as np
from scipy import stats
import os

# Definir las rutas de los archivos
input_file = '../datasets/preprocessed_data.csv'
output_file = '../datasets/preprocessed_data.csv'  # Sobrescribiremos el mismo archivo

# Leer el CSV
df = pd.read_csv(input_file)

# Obtener las últimas 13 columnas
last_13_columns = df.columns[-13:]

# Función para calcular R²
def calculate_r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2

# Lista para almacenar columnas a eliminar
columns_to_remove = set()

# Calcular R² para cada par de columnas
for i in range(len(last_13_columns)):
    for j in range(i+1, len(last_13_columns)):
        col1 = last_13_columns[i]
        col2 = last_13_columns[j]
        r_squared = calculate_r_squared(df[col1], df[col2])
        
        if r_squared > 1 - 1e-9:
            print(f"Alto R² ({r_squared:.12f}) encontrado entre {col1} y {col2}")
            columns_to_remove.add(col2)  # Eliminar la última columna del par

# Eliminar las columnas identificadas
if columns_to_remove:
    print(f"Eliminando las siguientes columnas: {', '.join(columns_to_remove)}")
    df = df.drop(columns=columns_to_remove)
else:
    print("No se encontraron columnas para eliminar.")

# Guardar el DataFrame actualizado
df.to_csv(output_file, index=False)
print(f"Archivo actualizado guardado en: {output_file}")