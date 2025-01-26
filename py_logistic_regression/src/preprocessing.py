import pandas as pd
import json
import os

def preprocess_data(input_file='../datasets/dataset_train.csv'):
    """
    Preprocess the input dataset and save the result to a new CSV file.

    Parameters:
    input_file (str): Path to the input CSV file.
    
    The preprocessing steps include:
    - Reading column to drop from preprocessing_config.json if exists
    - Dropping specified column from correlation analysis
    - Dropping rows with missing data
    - Removing duplicate rows
    - Converting 'Birthday' to datetime format
    - Converting 'Best_Hand' to a binary variable (0 for Left, 1 for Right)
    - Resetting the index
    """
    # Output file name
    output_file = '../datasets/dataset_preprocessed.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Check if there's a column to drop from correlation analysis
    config_file = '../output/preprocessing_config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            column_to_drop = config.get('column_to_drop')
            if column_to_drop and column_to_drop in df.columns:
                print(f"Dropping column '{column_to_drop}' based on correlation analysis")
                df = df.drop(columns=[column_to_drop])

    # Drop rows with missing data
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert 'Birthday' to datetime format
    df['Birthday'] = pd.to_datetime(df['Birthday'])

    # Convert 'Best Hand' to a binary variable (0 for Left, 1 for Right)
    df['Best Hand'] = df['Best Hand'].map({'Left': 0, 'Right': 1})

    # Reset the index to a continuous sequence
    df = df.reset_index(drop=True)

    # Remove the original 'Index' column if it exists
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)

    # Save the preprocessed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f'''Preprocessing steps completed:
    \t- Read the file {input_file}
    \t- Checked for correlation-based column removal
    \t- Dropped rows with missing data
    \t- Removed duplicate rows
    \t- Converted 'Birthday' to datetime format
    \t- Converted 'Best Hand' to a binary variable (0 for Left, 1 for Right)
    \t- Reset the index and dropped the original 'Index' column if present
    \t- Saved the preprocessed DataFrame to: {output_file}
    ''')

if __name__ == "__main__":
    preprocess_data()