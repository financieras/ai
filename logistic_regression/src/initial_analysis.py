import aux.colors as c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def initial_exploration(input_file='../datasets/dataset_train.csv'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, index_col=0)
    # index_col=0 indica que la primera columna del archivo CSV contiene el índice del DataFrame

    # Print the data info
    print(f"\n{c.BLUE}Info about the file: {input_file}{c.RESET}\n")
    df.info()

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    # Show the first registers in the DataFrame
    print(f"\n{c.BLUE}The first registers in the DataFrame{c.RESET}\n")
    print(df.head(10))

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    # Show the Null values by column
    print(f"\n{c.BLUE}Null values by column{c.RESET}\n")
    print(df.isnull().sum())

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    # Show the unique values by nominal features ('Hogwarts House' and 'Best Hand')
    print(f"\n{c.BLUE}Unique values by categoric feature{c.RESET}\n")

    # Analysis of unique values for 'Hogwarts House'
    print("Unique values in Hogwarts House:")
    print(df['Hogwarts House'].value_counts())

    # Analysis of unique values for 'Best Hand'
    print("\nUnique values in Best Hand:")
    print(df['Best Hand'].value_counts())


if __name__ == "__main__":
    initial_exploration()