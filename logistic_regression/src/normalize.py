import pandas as pd
import numpy as np
from datetime import datetime

# Define constant sets of features
BASE_FEATURES = [
    'Best Hand', 'Age', 'House_Gryffindor', 'House_Hufflepuff', 
    'House_Ravenclaw', 'House_Slytherin'
]

LITE5_COURSES = [
    'Herbology', 'Defense Against the Dark Arts', 'Potions', 
    'Charms', 'Flying'
]

LITE8_ADDITIONAL = [
    'Divination', 'Transfiguration', 'Care of Magical Creatures'
]

def normalize_data(input_file='../datasets/dataset_preprocessed.csv'):
    """
    Normalize the numerical columns of a given CSV file using mean and standard deviation.
    Creates three normalized CSV files with different feature sets:
    - Full dataset (all float64 columns)
    - Lite8 dataset (8 courses + 6 base features)
    - Lite5 dataset (5 courses + 6 base features)

    Parameters:
    input_file (str): The name of the input CSV file to be normalized (default is 'dataset_preprocessed.csv')
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

    # Convert int values in 'Best Hand' column to float64 values
    df['Best Hand'] = df['Best Hand'].astype(float)

    # Save the complete normalized dataset
    output_file = '../datasets/dataset_normalized.csv'
    df.to_csv(output_file, index=False)
    print(f"Normalization completed. Data saved in '{output_file}'")

    # Create lite8 and lite5 column lists
    lite5_columns = BASE_FEATURES + LITE5_COURSES
    lite8_columns = lite5_columns + LITE8_ADDITIONAL
    
    # Create and save lite8 version
    df_lite8 = df[lite8_columns]
    output_file_lite8 = '../datasets/dataset_normalized_lite8.csv'
    df_lite8.to_csv(output_file_lite8, index=False)
    print(f"Lite8 normalization completed. Data saved in '{output_file_lite8}'")
    
    # Create and save lite5 version
    df_lite5 = df[lite5_columns]
    output_file_lite5 = '../datasets/dataset_normalized_lite5.csv'
    df_lite5.to_csv(output_file_lite5, index=False)
    print(f"Lite5 normalization completed. Data saved in '{output_file_lite5}'")

    # Print information about the datasets
    print("\nDatasets summary:")
    
    print("\nCommon features across all datasets:")
    for i, feature in enumerate(BASE_FEATURES):
        print(f"\t{i+1}. {feature}")

    print(f"\nLite5 dataset: {len(df_lite5.columns)} features")
    print("The Hogwarts courses considered in Lite5:")
    for i, course in enumerate(LITE5_COURSES, 1):
        print(f"\t{i}. {course}")

    print(f"\nLite8 dataset: {len(df_lite8.columns)} features")
    print("Additional courses considered in Lite8:")
    for i, course in enumerate(LITE8_ADDITIONAL, 1):
        print(f"\t{i}. {course}")

    print(f"\nComplete dataset: {len(df.columns)} features")
    print("Additional courses considered in the complete dataset:")
    additional_courses = [col for col in df.columns if col not in lite8_columns]
    for i, course in enumerate(additional_courses, 1):
        print(f"\t{i}. {course}")

if __name__ == "__main__":
    normalize_data()