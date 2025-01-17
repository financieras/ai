import pandas as pd
from tabulate import tabulate
from aux.ft_functions import (
    ft_count, ft_mean, ft_std, ft_min, ft_max,
    ft_percentile, ft_median, ft_iqr,
    ft_skewness, ft_kurtosis, ft_cv
)

def calculate_metrics(df):
    """Calculate metrics for float columns."""
    # Select numeric columns (float64)
    numeric_columns = df.select_dtypes(include=['float64']).columns
    metrics = {}
    
    for col in numeric_columns:
        values = df[col].dropna().tolist()
        metrics[col] = {
            "Count": ft_count(values),
            "Mean": ft_mean(values),
            "Std": ft_std(values),
            "Min": ft_min(values),
            "25%": ft_percentile(values, 0.25),
            "50%": ft_median(values),
            "75%": ft_percentile(values, 0.75),
            "Max": ft_max(values),
            "IQR": ft_iqr(values),
            "Skewness": ft_skewness(values),
            "Kurtosis": ft_kurtosis(values),
            "CV": ft_cv(values)
        }
    
    return metrics

def print_metrics_table(metrics):
    """Print calculated metrics in a formatted table."""
    table_data = []
    headers = [""] + list(metrics.keys())
    
    metrics_to_display = [
        "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max",
        "IQR", "Skewness", "Kurtosis", "CV"
    ]
    
    for metric in metrics_to_display:
        row = [metric]
        for col in metrics:
            value = metrics[col][metric]
            row.append(f"{value:.6f}" if isinstance(value, float) else f"{value}")
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

def analyze_dataset(file_path='../datasets/dataset_preprocessed.csv'):
    """Analyze dataset by loading and calculating metrics."""
    try:
        df = pd.read_csv(file_path)
        metrics = calculate_metrics(df)
        print_metrics_table(metrics)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found")
    except Exception as e:
        print(f"Error reading the file: {str(e)}")

if __name__ == "__main__":
    analyze_dataset()