import pandas as pd
import numpy as np
import os

def normalize_data(input_file='preprocessed_data.csv', output_file='normalized_data.csv'):
    """
    Normalize the last 12 columns of a given CSV file using mean and standard deviation.

    Parameters:
    input_file (str): The name of the input CSV file to be normalized (default is 'preprocessed_data.csv').
    output_file (str): The name of the output CSV file to save the normalized data (default is 'normalized_data.csv').

    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    
    # Build complete paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'datasets', input_file)
    output_path = os.path.join(base_path, 'datasets', output_file)

    # Load the dataset
    df = pd.read_csv(input_path)

    # Identify the last 12 columns
    columns_to_normalize = df.columns[-12:]

    # Function to normalize using mean and standard deviation
    def normalize(column):
        mean = column.mean()
        std = column.std()
        return (column - mean) / std

    # Apply normalization to the last 12 columns
    df[columns_to_normalize] = df[columns_to_normalize].apply(normalize)

    # Save the normalized dataset
    df.to_csv(output_path, index=False)

    print(f"Normalization completed. Data saved in '{output_file}'.")

    return df  # Optional: return the normalized DataFrame

if __name__ == "__main__":
    # This block will execute only if the script is run directly
    normalize_data()