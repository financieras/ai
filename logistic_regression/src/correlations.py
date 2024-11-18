import pandas as pd
import numpy as np
from scipy import stats
import os

def remove_highly_correlated_columns(input_file='../datasets/preprocessed_data.csv', output_file='../datasets/preprocessed_data.csv'):
    """
    Removes columns with high correlation from the input dataset and saves the result.

    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the processed CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Get the last 13 columns
    last_13_columns = df.columns[-13:]

    # Function to calculate R²
    def calculate_r_squared(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return r_value**2

    # List to store columns to remove
    columns_to_remove = set()

    # Calculate R² for each pair of columns
    for i in range(len(last_13_columns)):
        for j in range(i+1, len(last_13_columns)):
            col1 = last_13_columns[i]
            col2 = last_13_columns[j]
            r_squared = calculate_r_squared(df[col1], df[col2])
            
            if r_squared > 1 - 1e-9:
                print(f"High R² ({r_squared:.12f}) found between {col1} and {col2}")
                columns_to_remove.add(col2)  # Remove the last column of the pair

    # Drop the identified columns
    if columns_to_remove:
        print(f"Proposed Columns to remove: {', '.join(columns_to_remove)}")
        #df = df.drop(columns=columns_to_remove)
    else:
        print("No columns found to remove.")

    # Save the updated DataFrame
    df.to_csv(output_file, index=False)
    print(f"Updated file saved to: {output_file}")