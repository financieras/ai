import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads the dataset from a CSV file and adds column names.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded dataset with named columns.
    """
    # Define column names
    column_names = ['id', 'diagnosis'] + [f'feature{i:02d}' for i in range(1, 31)]
    
    # Load the data without header
    data = pd.read_csv(file_path, header=None, names=column_names)
    
    print("Columnas en el conjunto de datos:")
    print(data.columns)
    
    return data

def clean_data(data):
    """Cleans the data by removing missing values and outliers.

    Args:
        data (pandas.DataFrame): The dataset to clean.

    Returns:
        pandas.DataFrame: The cleaned dataset.
    """
    # Remove rows with missing values
    data.dropna(inplace=True)
    
    # Convert 'diagnosis' to numeric (0 for 'B', 1 for 'M')
    data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
    
    return data

def normalize_numerical_features(data):
    """Normalizes numerical features using StandardScaler.

    Args:
        data (pandas.DataFrame): The dataset to normalize.

    Returns:
        pandas.DataFrame: The dataset with normalized numerical features.
    """
    scaler = StandardScaler()
    numerical_features = [col for col in data.columns if col.startswith('feature')]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

def main():
    # Load the data
    data = load_data('data/data.csv')
    
    print("\nInformación del dataset original:")
    print(data.info())
    
    # Preprocess the data
    data = clean_data(data)
    data = normalize_numerical_features(data)
    
    print("\nInformación del dataset procesado:")
    print(data.info())
    
    # Save the preprocessed data
    data.to_csv('data/preprocessed_data.csv', index=False)
    print("\nDatos preprocesados guardados en 'data/preprocessed_data.csv'")

if __name__ == "__main__":
    main()