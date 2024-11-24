import pandas as pd
import scipy.stats as stats
from tabulate import tabulate

def calculate_metrics(df):
    """Calculate metrics for float columns."""
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
            "Max": df[col].max(),
            "IQR": df[col].quantile(0.75) - df[col].quantile(0.25),
            "Skewness": df[col].skew(),                     # Asimetría
            "Kurtosis": df[col].kurtosis(),                 # Kurtosis
            "CV": abs(df[col].std() / df[col].mean())       # Coefficient of Variation
        }
    
    return metrics

def print_metrics_table(metrics):
    """Print calculated metrics in a formatted table."""
    table_data = []
    headers = [""] + list(metrics.keys())
    
    metrics_to_display = [
        "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max",
        "IQR", "Skewness", "Kurtosis", "CV"]
    
    for metric in metrics_to_display:
        row = [metric]
        for col in metrics:
            value = metrics[col][metric]
            row.append(f"{value:.6f}" if isinstance(value, float) else f"{value}")
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

def analyze_dataset(file_path='../datasets/preprocessed_data.csv'):
    """Analyze dataset by loading and calculating metrics."""
    df = pd.read_csv(file_path)
    metrics = calculate_metrics(df)
    print_metrics_table(metrics)

if __name__ == "__main__":
    analyze_dataset()