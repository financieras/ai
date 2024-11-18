import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # Read the dataset
    df = pd.read_csv('ai/logistic_regression/datasets/dataset_train.csv')
    
    # Get list of course columns
    '''courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Divination', 'Muggle_Studies',
              'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
              'Care_of_Magical_Creatures', 'Charms', 'Flying']
    '''

    courses = df.columns[-13:]

    # Calculate correlation matrix
    correlation_matrix = df[courses].corr()
    
    # Find the two most correlated features
    # Get upper triangle of correlation matrix
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Find the maximum correlation value and its position
    max_corr = upper_tri.max().max()
    feature1, feature2 = np.where(upper_tri == max_corr)
    feature1, feature2 = courses[feature1[0]], courses[feature2[0]]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=feature1, y=feature2, hue='Hogwarts_House', alpha=0.6)
    
    # Add correlation line
    z = np.polyfit(df[feature1], df[feature2], 1)
    p = np.poly1d(z)
    plt.plot(df[feature1], p(df[feature1]), "r--", alpha=0.8)
    
    # Add correlation coefficient to title
    plt.title(f'Correlation between {feature1} and {feature2}\nCorrelation coefficient: {max_corr:.3f}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
    # Print the correlation value
    print(f"The most similar features are {feature1} and {feature2}")
    print(f"Their correlation coefficient is: {max_corr:.3f}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('scatter_plot.png')
    plt.close()

if __name__ == "__main__":
    main()

'''
Este script:
1. Lee el dataset
2. Calcula la matriz de correlación entre todas las materias
3. Encuentra automáticamente las dos materias que tienen la correlación más alta
4. Crea un scatter plot de estas dos materias
5. Añade una línea de tendencia
6. Muestra el coeficiente de correlación
7. Guarda el resultado en 'scatter_plot.png'

La correlación va de -1 a 1, donde:
- 1 indica una correlación positiva perfecta
- -1 indica una correlación negativa perfecta
- 0 indica ninguna correlación

El script encontrará automáticamente las dos materias que tienen la correlación más alta (más cercana a 1 o -1), lo que nos indicará qué dos características son más similares en términos de sus patrones de puntuación.

Al ejecutar el script, mostrará en la consola cuáles son las dos materias más similares y su coeficiente de correlación, además de generar el gráfico que lo visualiza.
'''
