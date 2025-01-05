import aux.colors as c
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg for interactive plotting
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def detect_highly_correlated_columns(input_file="../datasets/dataset_train.csv"):
    """
    Detect and visualize highly correlated columns in a dataset.

    This function loads a CSV file, calculates correlations between numeric columns,
    creates a heatmap, and identifies highly correlated pairs of columns.

    Args:
    input_file (str): Path to the input CSV file. Default is "../datasets/dataset_train.csv".

    Returns:
    None. Outputs are saved as files and printed to console.
    """
    CORRELATION_THRESHOLD = 0.6   # Cut-off Correlation
    output_image_file = '../output/correlation_heatmap.png'
    output_csv_file = '../output/high_correlations.csv'

    # Load the dataset
    df = pd.read_csv(input_file, index_col=0)

    ### Search very high correlations ###
    print(f"\n{c.BLUE}Search very high correlations{c.RESET}\n")

    # Select numeric columns (float64)
    numeric_columns = df.select_dtypes(include=['float64']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Create a heatmap of correlations and save it
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_image_file)
    plt.close()

    # Find high correlations (in absolute value)
    high_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):  # exclude autocorrelations
            if abs(correlation_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:  # threshold
                high_correlations.append((correlation_matrix.index[i], 
                                        correlation_matrix.columns[j], 
                                        correlation_matrix.iloc[i, j]))

    # Sort high correlations by absolute value
    high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f'''The detect highly correlated columns steps include:\n
        \t- Read the file {input_file}
        \t- Select numeric columns (float64, int64, and bool)
        \t- Calculate the correlation matrix
        \t- Create a heatmap of correlations
        \t- Find high correlations (in absolute value)
        \t- Sort high correlations by absolute value
        \t- Print high correlations (|r| > {CORRELATION_THRESHOLD})
        \t- Save the heatmap of correlations to a image file: {output_image_file}
        \t- Print high correlations table
        \t- Save high correlations to a CSV file: {output_csv_file}
        \t- Show heatmap image
        ''')

    # Print high correlations
    print(f"High correlations (|r| > {CORRELATION_THRESHOLD}):\n")
    table_data = []
    for var1, var2, corr in high_correlations:
        table_data.append([var1, var2, round(corr, 2)])
    print(tabulate(table_data, tablefmt="fancy_grid"))

    # Save high correlations to a CSV file
    correlations_df = pd.DataFrame(high_correlations, columns=['Variable 1', 'Variable 2', 'Correlation'])
    correlations_df.to_csv(output_csv_file, index=False)

    # Heatmap image and high correlations inform have been saved in the 'output' folder
    print(f"\nHeatmap image and high correlations inform have been saved in the 'output' folder.")
    
    # Ask user if they want to see the heatmap
    while True:
        response = input("\nWould you like to see the correlation image? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            # Show heatmap image using matplotlib
            img = plt.imread(output_image_file)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            break
        elif response in ['n', 'no']:
            print("The graph will not be displayed. You can find it at:", output_image_file)
            break
        else:
            print("Please answer 'y' or 'n'")

if __name__ == "__main__":
    detect_highly_correlated_columns()