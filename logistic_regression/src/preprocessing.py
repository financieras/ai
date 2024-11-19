import numpy as np
import pandas as pd
import os

def preprocess_data(input_file='../datasets/dataset_train.csv', output_file='../datasets/preprocessed_data.csv'):
    """
    Preprocess the input dataset and save the result to a new file.

    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the preprocessed CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Drop rows with missing data
    df = df.dropna()

    # Reset the index to a continuous sequence
    sequential_index = np.arange(len(df))
    df['Index'] = sequential_index
    # Rename columns, replacing spaces with underscores
    df.columns = df.columns.str.replace(' ', '_')

    # Round 'Divination' and 'Charms' to 12 decimal places
    df['Divination'] = df['Divination'].round(12)
    df['Charms'] = df['Charms'].round(12)

    # Convert 'Best_Hand' to a binary variable
    df['Best_Hand'] = df['Best_Hand'].map({'Left': 0, 'Right': 1})

    # Save the preprocessed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Preprocessing completed. Output saved to: {output_file}")