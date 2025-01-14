import pandas as pd
import numpy as np
from datetime import datetime

def normalize_data(input_file='../datasets/dataset_preprocessed.csv', output_file='../datasets/dataset_normalized.csv'):
    """
    Normalize the numerical columns of a given CSV file using mean and standard deviation.
    Also calculates age from Birthday column and removes unnecessary columns.
    The normalized data is saved to a new CSV file.

    Parameters:
    input_file (str): The name of the input CSV file to be normalized (default is 'dataset_preprocessed.csv').
    output_file (str): The name of the output CSV file to save the normalized data (default is 'dataset_normalized.csv').
    """
    
    # Read the dataset
    df = pd.read_csv(input_file)

    # Calculate age based on the maximum date in Birthday column
    df['Birthday'] = pd.to_datetime(df['Birthday'])
    reference_date = df['Birthday'].max()
    df['Age'] = (reference_date - df['Birthday']).dt.days / 365.25

    # Remove unnecessary columns
    columns_to_drop = ['First Name', 'Last Name', 'Birthday']
    df = df.drop(columns=columns_to_drop)
    
    # Select numerical columns to normalize (including Age which is already float64)
    columns_to_normalize = df.select_dtypes(include=['float64']).columns.tolist()

    # Function to normalize using mean and standard deviation
    def normalize(column):
        mean = column.mean()
        std = column.std()
        return (column - mean) / std

    # Apply normalization to the selected columns
    df[columns_to_normalize] = df[columns_to_normalize].apply(normalize)

    # Apply one-hot encoding for Hogwarts House
    df = pd.get_dummies(df, columns=['Hogwarts House'], prefix='House', dtype=float)

    # Convert int values in 'Best Hand' column in float64 values
    df['Best Hand'] = df['Best Hand'].astype(float)

    # Save the normalized dataset
    df.to_csv(output_file, index=False)

    print(f"Normalization completed. Data saved in '{output_file}'.")

    # Columns in DataFrame
    print("\nColumns in DataFrame:\n")
    columns = df.columns.tolist()
    for i, column in enumerate(columns, start=1):
        print(f"{i}. {column}")


if __name__ == "__main__":
    normalize_data()