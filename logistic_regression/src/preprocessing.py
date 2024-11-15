import numpy as np
import pandas as pd
import os               # para manejar rutas de archivos

# Definir las rutas de los archivos
input_file = '../datasets/dataset_train.csv'
output_file = '../datasets/preprocessed_data.csv'

# Asegurarse de que la ruta de salida existe
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Leer el CSV
df = pd.read_csv(input_file)

# Eliminar filas con datos faltantes
df = df.dropna()

# Renombrar columnas reemplazando espacios por guiones bajos
df.columns = df.columns.str.replace(' ', '_')

# Redondear 'Divination' y 'Charms' a 12 decimales
df['Divination'] = df['Divination'].round(12)
df['Charms'] = df['Charms'].round(12)

# Resetear el índice para que sea continuo
# Crear un array de números correlativos
index_values = np.arange(len(df))

# Asignar el array a la columna "Index"
df['Index'] = index_values

# Convertir 'Best_Hand' a variable binaria
df['Best_Hand'] = df['Best_Hand'].map({'Left': 0, 'Right': 1})

# Guardar el DataFrame procesado en un nuevo archivo CSV
df.to_csv(output_file, index=False)

print(f"Preprocesamiento completado. Archivo guardado en: {output_file}")