import pandas as pd
import numpy as np
import sys
from tabulate import tabulate

def load_data(file_path):
    """Carga el dataset desde el archivo CSV."""
    return pd.read_csv(file_path)

def get_sample(df, sample_size=None):
    """Obtiene una muestra aleatoria del dataframe o el dataframe completo."""
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=42)

def calculate_metrics(df):
    """Calcula métricas para las 13 últimas columnas numéricas del dataframe."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns[-13:]
    metrics = {}
    
    for col in numeric_columns:
        metrics[col] = {
            "Count": df[col].count(),
            "Mean": df[col].mean(),
            "Std": df[col].std(),
            "Min": df[col].min(),
            "25%": df[col].quantile(0.25),
            "50%": df[col].median(),
            "75%": df[col].quantile(0.75),
            "Max": df[col].max()
        }
    
    return metrics

def print_metrics_table(metrics):
    """Imprime las métricas calculadas en formato de tabla."""
    table_data = []
    headers = [""] + list(metrics.keys())
    
    for metric in ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]:
        row = [metric]
        for col in metrics:
            value = metrics[col][metric]
            row.append(f"{value:.6f}" if isinstance(value, float) else f"{value}")
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

def main():
    if len(sys.argv) < 2:
        print("Uso: python description.py <archivo_csv> [tamaño_muestra]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Verifica si se proporcionó un argumento para el tamaño de la muestra
    sample_size = None
    if len(sys.argv) > 2:
        try:
            sample_size = int(sys.argv[2])
        except ValueError:
            print("El tamaño de la muestra debe ser un número entero.")
            sys.exit(1)
    
    # Carga los datos
    df = load_data(file_path)
    
    # Obtiene la muestra o el conjunto completo
    sample_df = get_sample(df, sample_size)
    
    # Calcula las métricas
    metrics = calculate_metrics(sample_df)
    
    # Imprime las métricas en formato de tabla
    print_metrics_table(metrics)

if __name__ == "__main__":
    main()