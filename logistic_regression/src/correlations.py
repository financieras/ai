import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def detect_highly_correlated_columns(input_file="../datasets/preprocessed_data.csv"):
    """
    Detect and visualize highly correlated columns in a dataset.

    This function loads a CSV file, calculates correlations between numeric columns,
    creates a heatmap, and identifies highly correlated pairs of columns.

    Args:
    input_file (str): Path to the input CSV file. Default is "../datasets/preprocessed_data.csv".

    Returns:
    None. Outputs are saved as files and printed to console.
    """

    # Load the dataset
    df = pd.read_csv(input_file, index_col=0)

    # Select numeric columns (float64, int64, and bool)
    numeric_columns = df.select_dtypes(include=['float64', 'int64', 'bool']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Create a heatmap of correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../output/correlation_heatmap.png')
    plt.close()

    # Find high correlations (in absolute value)
    high_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.80:  # threshold
                high_correlations.append((correlation_matrix.index[i], 
                                        correlation_matrix.columns[j], 
                                        correlation_matrix.iloc[i, j]))

    # Sort high correlations by absolute value
    high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    # Print high correlations
    print("High correlations (|r| > 0.80):")
    for var1, var2, corr in high_correlations:
        print(f"{var1} - {var2}: {corr:.2f}")

    # Save high correlations to a CSV file
    correlations_df = pd.DataFrame(high_correlations, columns=['Variable 1', 'Variable 2', 'Correlation'])
    correlations_df.to_csv('../output/high_correlations.csv', index=False)

    print("\nHeatmap and high correlations have been saved in the 'output' folder.")

if __name__ == "__main__":
    detect_highly_correlated_columns()