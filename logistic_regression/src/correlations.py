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
        response = input(f"\n{c.CYAN}Would you like to see the correlation image? (y/n): {c.RESET}").lower().strip()
        if response in ['y', 'yes']:
            # Show heatmap image using matplotlib
            img = plt.imread(output_image_file)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            break
        elif response in ['n', 'no']:
            print("\nThe graph will not be displayed. You can find it at:", output_image_file)
            break
        else:
            print("Please answer 'y' or 'n'")

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")


    # Drop the feature with a perfect correlation

    if round(abs(high_correlations[0][2]), 10) == 1:
        print(f"\n{c.BLUE}Drop the feature with a perfect correlation{c.RESET}\n")
        print("There is a perfect correlation between two features.")
        print("To properly train the neural network, we will remove one of them.")
        print("The one that leaves the highest number of complete records will remain.")
        feature1 = high_correlations[0][0]
        feature2 = high_correlations[0][1]
        # Which column to delete?
        print(f"\n{c.BLUE}Which feature should I drop to avoid multicollinearity?{c.RESET}\n")
        print(f"\t{c.GREEN}1. {feature1} or\n")
        print(f"\t2. {feature2}{c.RESET}\n")

        # Function to count complete records when deleting a column
        def count_complete_records(df, column_to_drop):
            df_temp = df.drop(columns=[column_to_drop])
            return df_temp.dropna().shape[0]

        # Count full records when removing Feature1
        records_without_feature1 = count_complete_records(df,feature1)
        print(f"Complete records after removing {feature1}: {records_without_feature1}")

        # Count full records when removing Feature2
        records_without_feature2 = count_complete_records(df, feature2)
        print(f"Complete records after removing {feature2}: {records_without_feature2}")

        # Determine which feature to remove
        if records_without_feature1 >= records_without_feature2:
            feature_to_drop = feature1
            records_kept = records_without_feature1
        else:
            feature_to_drop = feature2
            records_kept = records_without_feature2

        print(f"It is recommended to {c.GREEN}remove {feature_to_drop}{c.RESET} to maintain {records_kept} complete records.")


if __name__ == "__main__":
    detect_highly_correlated_columns()