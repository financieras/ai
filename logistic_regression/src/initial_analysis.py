import aux.colors as c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def initial_exploration(input_file='../datasets/dataset_train.csv'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, index_col=0)
    # index_col=0 indica que la primera columna del archivo CSV contiene el índice del DataFrame

    print(f"\n{c.BLUE}Info about the file: {input_file}{c.RESET}\n")
    df.info()

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    # Show the first rows
    print(df.head(10))

    # Null values by column
    print(f"\n{c.BLUE}Null values by column{c.RESET}\n")
    print(df.isnull().sum())

    # Analysis of unique values for 'Hogwarts House'
    print("\nUnique values in Hogwarts House:")
    print(df['Hogwarts House'].value_counts())

    # Analysis of unique values for 'Best Hand'
    print("\nUnique values in Best Hand:")
    print(df['Best Hand'].value_counts())

    # Search correlation 1 or -1
    
    # Select numeric columns (float64)
    numeric_columns = df.select_dtypes(include=['float64']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Find high correlations (in absolute value)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            coef_correl = correlation_matrix.iloc[i, j]
            if abs(coef_correl) > 0.99:
                correlation_matrix.index[i], correlation_matrix.columns[j]
                print(f"A high correlation ratio has been found between these two features:")
                print(f"- {correlation_matrix.index[i]}")
                print(f"- {correlation_matrix.index[j]}")
                print(f"Coeficient of correlation: {coef_correl}")


if __name__ == "__main__":
    initial_exploration()