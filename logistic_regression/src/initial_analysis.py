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

    # Show the first 
    print(f"\n{c.BLUE}The 10 first registers in the DataFrame{c.RESET}\n")
    print(df.head(10))

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    # Null values by column
    print(f"\n{c.BLUE}Null values by column{c.RESET}\n")
    print(df.isnull().sum())

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    print(f"\n{c.BLUE}Unique values by categoric feature{c.RESET}\n")

    # Analysis of unique values for 'Hogwarts House'
    print("Unique values in Hogwarts House:")
    print(df['Hogwarts House'].value_counts())

    # Analysis of unique values for 'Best Hand'
    print("\nUnique values in Best Hand:")
    print(df['Best Hand'].value_counts())

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    ### Search very high correlations ###
    print(f"\n{c.BLUE}Search very high correlations{c.RESET}\n")
    
    # Select numeric columns (float64)
    numeric_columns = df.select_dtypes(include=['float64']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Find high correlations (in absolute value)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):  # exclude autocorrelations
            coef_correl = correlation_matrix.iloc[i, j]     # iloc = index location
            if abs(coef_correl) > 0.9:                      # only interested in very high correlations
                feature1 = correlation_matrix.index[i]
                feature2 = correlation_matrix.index[j]
                print(f"A high correlation ratio has been found between these two features:")
                print(f"- {feature1}")
                print(f"- {feature2}")
                print(f"Coeficient of correlation: {coef_correl}")

    input(f"\n{c.YELLOW}Press ENTER to continue...{c.RESET}")

    # ¿Qué columna eliminar?
    print(f"\n{c.BLUE}¿Qué columna eliminar? {feature1} or {feature2}{c.RESET}\n")
    
    # Función para contar registros completos al eliminar una columna
    def count_complete_records(df, column_to_drop):
        df_temp = df.drop(columns=[column_to_drop])
        return df_temp.dropna().shape[0]

    # Contar registros completos al eliminar Feature1
    records_without_feature1 = count_complete_records(df,feature1)
    print(f"Registros completos al eliminar {feature1}: {records_without_feature1}")

    # Contar registros completos al eliminar Feature2
    records_without_feature2 = count_complete_records(df, feature2)
    print(f"Registros completos al eliminar {feature2}: {records_without_feature2}")

    # Determinar qué característica eliminar
    if records_without_feature1 >= records_without_feature2:
        feature_to_drop = feature1
        records_kept = records_without_feature1
    else:
        feature_to_drop = feature2
        records_kept = records_without_feature2

    print(f"Se recomienda eliminar {feature_to_drop} para mantener {records_kept} registros completos.")


if __name__ == "__main__":
    initial_exploration()