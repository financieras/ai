import pandas as pd

def preprocess_data(input_file='../datasets/dataset_train.csv', output_file='../datasets/preprocessed_data.csv'):
    """
    Preprocess the input dataset and save the result to a new CSV file.

    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the preprocessed CSV file.
    
    The preprocessing steps include:
    - Dropping rows with missing data
    - Removing duplicate rows
    - Converting 'Birthday' to datetime format
    - Renaming columns by replacing spaces with underscores
    - Converting 'Best_Hand' to a binary variable (0 for Left, 1 for Right)
    - Resetting the index and dropping the original 'Index' column if present.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Drop rows with missing data
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert 'Birthday' to datetime format
    df['Birthday'] = pd.to_datetime(df['Birthday'])

    # Rename columns by replacing spaces with underscores
    df.columns = df.columns.str.replace(' ', '_')

    # Convert 'Best_Hand' to a binary variable (0 for Left, 1 for Right)
    df['Best_Hand'] = df['Best_Hand'].map({'Left': 0, 'Right': 1})

    # Reset the index to a continuous sequence
    df = df.reset_index(drop=True)

    # Remove the original 'Index' column if it exists
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)

    # Save the preprocessed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f'''The preprocessing steps include:\n
    \t- Read the file {input_file}
    \t- Dropping rows with missing data
    \t- Removing duplicate rows
    \t- Converting 'Birthday' to datetime format
    \t- Renaming columns by replacing spaces with underscores
    \t- Converting 'Best_Hand' to a binary variable (0 for Left, 1 for Right)
    \t- Resetting the index and dropping the original 'Index' column if present.
    \t- Save the preprocessed DataFrame to a new CSV file.
        ''')
    print(f"Preprocessing completed. Output saved to: {output_file}")

    return df

if __name__ == "__main__":
    preprocess_data()