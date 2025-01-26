import aux.colors as c
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import json

def detect_highly_correlated_columns(input_file="../datasets/dataset_train.csv"):
    """
    Detect and visualize highly correlated columns in a dataset.
    
    Args:
    input_file (str): Path to the input CSV file.
    
    Returns:
    str: Name of the column to be dropped if perfect correlation is found, None otherwise.
    """
    CORRELATION_THRESHOLD = 0.6
    output_image_file = '../output/correlation_heatmap.png'

    # Load the dataset
    df = pd.read_csv(input_file, index_col=0)

    print(f"\n{c.BLUE}Search very high correlations{c.RESET}\n")
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['float64']).columns
    correlation_matrix = df[numeric_columns].corr()

    # Create and save heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_image_file)
    plt.close()

    # Find high correlations
    high_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
                high_correlations.append((correlation_matrix.index[i], 
                                       correlation_matrix.columns[j], 
                                       correlation_matrix.iloc[i, j]))

    # Sort by absolute correlation value
    high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    # Print high correlations
    print(f"High correlations (|r| > {CORRELATION_THRESHOLD}):\n")
    table_data = []
    for var1, var2, corr in high_correlations:
        table_data.append([var1, var2, round(corr, 2)])
    print(tabulate(table_data, tablefmt="fancy_grid"))

    # Ask to show heatmap
    while True:
        response = input(f"\n{c.CYAN}Would you like to see the correlation image? (y/n): {c.RESET}").lower().strip()
        if response in ['y', 'yes']:
            img = plt.imread(output_image_file)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            break
        elif response in ['n', 'no']:
            print(f"\nThe graph will not be displayed. You can find it at:\n\t{output_image_file}")
            break
        else:
            print("Please answer 'y' or 'n'")

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    # Handle perfect correlations
    column_to_drop = None
    if high_correlations and round(abs(high_correlations[0][2]), 10) == 1:
        print(f"\n{c.BLUE}Analyzing perfect correlation{c.RESET}\n")
        feature1 = high_correlations[0][0]
        feature2 = high_correlations[0][1]
        
        # Get all columns except the perfectly correlated ones
        all_columns = list(df.columns)
        rest_columns = [col for col in all_columns if col not in [feature1, feature2]]
        
        # Create masks for each feature
        mask_feature1 = df[feature1].isna() & df[rest_columns].notna().all(axis=1)
        mask_feature2 = df[feature2].isna() & df[rest_columns].notna().all(axis=1)
        
        data1 = mask_feature1.sum()
        data2 = mask_feature2.sum()
        
        # Determine which feature to drop
        if data1 > data2:
            column_to_drop = feature1
            print(f"Propose to remove '{feature1}' as we gain {data1 - data2} complete records.")
        else:
            column_to_drop = feature2
            print(f"Propose to remove '{feature2}' as we gain {data2 - data1} complete records.")
            
        # Save the column to drop in the output directory
        config_file = '../output/preprocessing_config.json'
        config = {'column_to_drop': column_to_drop}
        with open(config_file, 'w') as f:
            json.dump(config, f)
            
        print(f"\nColumn to be dropped has been saved in the file:\n\t{config_file}")
    
    return column_to_drop

if __name__ == "__main__":
    detect_highly_correlated_columns()