import pandas as pd
import numpy as np
import os

def normalize_data(input_file='../datasets/preprocessed_data.csv', output_file='../datasets/normalized_data.csv'):
    """
    Normalize the last 12 columns of a given CSV file using mean and standard deviation.

    Parameters:
    input_file (str): The name of the input CSV file to be normalized (default is 'preprocessed_data.csv').
    output_file (str): The name of the output CSV file to save the normalized data (default is 'normalized_data.csv').

    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    
    # Read the dataset
    df = pd.read_csv(input_file)

    # Apply one-hot encoding for 'Hogwarts_House'
    df = pd.get_dummies(df, columns=['Hogwarts_House'], prefix='House')

    # Select numerical columns of type float64
    columns_to_normalize = df.select_dtypes(include=['float64']).columns

    # Function to normalize using mean and standard deviation
    def normalize(column):
        mean = column.mean()
        std = column.std()
        return (column - mean) / std

    # Apply normalization to the float64 columns
    df[columns_to_normalize] = df[columns_to_normalize].apply(normalize)

    # Save the normalized dataset
    df.to_csv(output_file, index=False)

    print(f"Normalization completed. Data saved in '{output_file}'.")

if __name__ == "__main__":
    # This block will execute only if the script is run directly
    normalize_data()