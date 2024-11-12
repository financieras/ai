import pandas as pd
import numpy as np
import sys

def load_data(file_path):
    """Carga el dataset desde el archivo CSV."""
    return pd.read_csv(file_path)

def get_sample(df, sample_size=None):
    """Obtiene una muestra aleatoria del dataframe o el dataframe completo."""
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=42)

def calculate_metrics(df):
    """Calcula métricas para las columnas numéricas del dataframe."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    metrics = {}
    
    for col in numeric_columns:
        metrics[col] = {
            "media": df[col].mean(),
            "mediana": df[col].median(),
            "desviacion_estandar": df[col].std(),
            "minimo": df[col].min(),
            "maximo": df[col].max(),
            "percentil_25": df[col].quantile(0.25),
            "percentil_75": df[col].quantile(0.75)
        }
    
    return metrics

def print_metrics(metrics):
    """Imprime las métricas calculadas."""
    for col, col_metrics in metrics.items():
        print(f"\nMétricas para {col}:")
        for metric, value in col_metrics.items():
            print(f"  {metric}: {value:.2f}")

def main():
    file_path = "../datasets/dataset_train.csv"
    
    # Verifica si se proporcionó un argumento para el tamaño de la muestra
    sample_size = None
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print("El argumento debe ser un número entero.")
            sys.exit(1)
    
    # Carga los datos
    df = load_data(file_path)
    
    # Obtiene la muestra o el conjunto completo
    sample_df = get_sample(df, sample_size)
    
    # Calcula las métricas
    metrics = calculate_metrics(sample_df)
    
    # Imprime las métricas
    print(f"Métricas calculadas para {'una muestra de ' + str(sample_size) if sample_size else 'todo el conjunto de datos'}:")
    print_metrics(metrics)

if __name__ == "__main__":
    main()