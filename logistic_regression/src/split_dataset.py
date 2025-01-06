import pandas as pd
import numpy as np

def split_stratified_dataset(df, train_ratio=0.7, validation_ratio=0.15, random_state=42):
    """
    Split dataframe maintaining house proportions
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Create empty dataframes for each split
    train_data = []
    validation_data = []
    test_data = []
    
    # Process each house separately to maintain proportions
    for house in ['House_Gryffindor', 'House_Hufflepuff', 'House_Ravenclaw', 'House_Slytherin']:
        # Get rows for current house
        house_data = df[df[house] == 1].copy()
        total_house = len(house_data)
        
        # Calculate sizes for each split
        train_size = int(total_house * train_ratio)
        validation_size = int(total_house * validation_ratio)
        
        # Shuffle the house data
        shuffled_indices = np.random.permutation(house_data.index)
        house_data = house_data.loc[shuffled_indices]
        
        # Split the data
        train_data.append(house_data.iloc[:train_size])
        validation_data.append(house_data.iloc[train_size:train_size + validation_size])
        test_data.append(house_data.iloc[train_size + validation_size:])
    
    # Combine all houses for each split
    train_df = pd.concat(train_data, axis=0)
    validation_df = pd.concat(validation_data, axis=0)
    test_df = pd.concat(test_data, axis=0)
    
    # Shuffle each split again to mix houses
    train_df = train_df.sample(frac=1, random_state=random_state)
    validation_df = validation_df.sample(frac=1, random_state=random_state)
    test_df = test_df.sample(frac=1, random_state=random_state)
    
    return train_df, validation_df, test_df

def split_dataset(input_file='../datasets/dataset_normalized.csv', 
                 train_ratio=0.7, 
                 validation_ratio=0.15,
                 random_state=42):
    """
    Split the normalized dataset into training, validation and test sets.
    The resulting files will be:
    - normal_dataset_train.csv
    - normal_dataset_validation.csv
    - normal_dataset_test.csv
    
    Parameters:
    input_file (str): Path to the normalized dataset
    train_ratio (float): Ratio for training set (default 0.7)
    validation_ratio (float): Ratio for validation set (default 0.15)
    random_state (int): Random seed for reproducibility
    """
    # Read the normalized dataset
    df = pd.read_csv(input_file)
    
    # Split the dataset
    train_df, validation_df, test_df = split_stratified_dataset(
        df, 
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        random_state=random_state
    )
    
    # Save the datasets
    train_df.to_csv('../datasets/normal_dataset_train.csv', index=False)
    validation_df.to_csv('../datasets/normal_dataset_validation.csv', index=False)
    test_df.to_csv('../datasets/normal_dataset_test.csv', index=False)
    
    # Print information about the splits
    print(f"Original dataset size: {len(df)}")
    print(f"Training set size: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Validation set size: {len(validation_df)} ({len(validation_df)/len(df):.1%})")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    # Print distribution of houses in each set
    def print_house_distribution(data, set_name):
        house_dist = {
            'Gryffindor': data['House_Gryffindor'].sum(),
            'Hufflepuff': data['House_Hufflepuff'].sum(),
            'Ravenclaw': data['House_Ravenclaw'].sum(),
            'Slytherin': data['House_Slytherin'].sum()
        }
        print(f"\n{set_name} house distribution:")
        for house, count in house_dist.items():
            print(f"{house}: {count} ({count/len(data):.1%})")
    
    print_house_distribution(train_df, "Training set")
    print_house_distribution(validation_df, "Validation set")
    print_house_distribution(test_df, "Test set")

if __name__ == "__main__":
    split_dataset()