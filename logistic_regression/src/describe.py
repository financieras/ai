import pandas as pd
import numpy as np
from tabulate import tabulate
import sys

def load_data(file_path):
    """Load the dataset from the CSV file."""
    return pd.read_csv(file_path)

def get_sample(df, sample_size=None):
    """Get a random sample of the dataframe or the full dataframe."""
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=42)

def calculate_metrics(df):
    """Calculate metrics for float columns in the dataframe."""
    # Select numeric columns (float64)
    numeric_columns = df.select_dtypes(include=['float64']).columns

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
    """Print the calculated metrics in a table format."""
    table_data = []
    headers = [""] + list(metrics.keys())
    
    for metric in ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]:
        row = [metric]
        for col in metrics:
            value = metrics[col][metric]
            row.append(f"{value:.6f}" if isinstance(value, float) else f"{value}")
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

def analyze_dataset(file_path='../datasets/preprocessed_data.csv', sample_size=None):
    """Analyze the dataset and print the metrics table."""
    df = load_data(file_path)
    sample_df = get_sample(df, sample_size)
    metrics = calculate_metrics(sample_df)
    print_metrics_table(metrics)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python describe.py <csv_file> [sample_size]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    analyze_dataset(file_path, sample_size)