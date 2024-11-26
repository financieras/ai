import pandas as pd
import numpy as np

def split_data(X, y, val_size, test_size, random_state=42):
    """
    Custom function to split data into train, validation and test sets using random sampling.
    
    Parameters:
    X (pd.DataFrame): Features
    y (pd.DataFrame): Target variables
    val_size (float): Proportion for validation set
    test_size (float): Proportion for test set
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Get total number of samples
    n_samples = len(X)
    
    # Calculate sizes for each set
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    # Create random permutation of indices
    all_indices = np.random.permutation(n_samples)
    
    # Randomly select indices for each set
    test_indices = all_indices[:n_test]
    val_indices = all_indices[n_test:n_test + n_val]
    train_indices = all_indices[n_test + n_val:]
    
    # Create the splits using the random indices
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_data(
    input_file='../datasets/preprocessed_data.csv',
    output_file='../datasets/normalized_data.csv',
    features=['Astronomy', 'Herbology', 'Potions', 'Charms', 'Flying'],
    test_size=0.15,
    validation_size=0.15,
    random_state=42
):
    """
    Prepares data for neural network training by filtering relevant columns, normalizing features,
    and splitting into train/validation/test sets. This function now includes data cleaning by
    removing unnecessary columns before processing.

    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    features (list): List of feature columns to use for training (only these and Hogwarts_House will be kept)
    test_size (float): Proportion of data to use for test set
    validation_size (float): Proportion of data to use for validation set
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test) where:
        - X sets contain only the specified features
        - y sets contain one-hot encoded house information
    """
    
    # Read the dataset
    df = pd.read_csv(input_file)

    # Keep only Hogwarts_House column and specified features
    columns_to_keep = ['Hogwarts_House'] + features

    # Remove all columns except those to keep
    df = df.drop(df.columns.drop(columns_to_keep), axis=1)

    # Apply one-hot encoding for 'Hogwarts_House' if not already encoded
    if 'Hogwarts_House' in df.columns:
        df = pd.get_dummies(df, columns=['Hogwarts_House'], prefix='House')

    # Select all float64 columns for normalization
    float_columns = df.select_dtypes(include=['float64']).columns
    
    # Normalize all float64 columns
    for column in float_columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std

    # Save the full normalized dataset
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Normalized data saved to '{output_file}'")

    # Select features and target variables for training
    X = df[features]
    y = df[['House_Gryffindor', 'House_Hufflepuff', 'House_Ravenclaw', 'House_Slytherin']]

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, validation_size, test_size, random_state
    )

    # Print dataset split information
    print("\nDataset split information:")
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\nFeatures used for training: {features}")
    print(f"All normalized columns: {list(float_columns)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Example usage
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()