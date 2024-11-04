import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads the dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Cleans the data by removing missing values and outliers.

    Args:
        data (pandas.DataFrame): The dataset to clean.

    Returns:
        pandas.DataFrame: The cleaned dataset.
    """
    # ... código para limpiar los datos ...
    # For example:
    # data.dropna(inplace=True)  # Remove rows with missing values
    # data = data[data['column_name'] < 10]  # Remove outliers
    return data

def normalize_numerical_features(data):
    """Normalizes numerical features using StandardScaler.

    Args:
        data (pandas.DataFrame): The dataset to normalize.

    Returns:
        pandas.DataFrame: The dataset with normalized numerical features.
    """
    scaler = StandardScaler()
    numerical_features = ['feature1', 'feature2']  # List of numerical features
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

# ... other functions? ...


if __name__ == "__main__":
    # Load the data
    data = load_data('data/data.csv')   # Relative path to the CSV file

    # Preprocess the data
    data = clean_data(data)
    data = normalize_numerical_features(data)

    # ... other preprocessing operations ...

    # Save the preprocessed data
    data.to_csv('preprocessed_data.csv', index=False)