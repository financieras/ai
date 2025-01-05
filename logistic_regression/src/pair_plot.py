import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def exploring_feature_relationships():
    # Cargar los datos
    df = pd.read_csv('../datasets/dataset_preprocessed.csv')

    # Get list of course columns (excluding non-course columns)
    courses = df.select_dtypes(include=['float64'])
    
    # Crear el pair plot
    sns.pairplot(df, hue='Hogwarts House', vars=courses)

    #plt.show()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('../output/pair_plot.png')
    plt.close()

    print("\nThe Pair Plot graph has been saved in the 'output' folder.")

if __name__ == "__main__":
    exploring_feature_relationships()

'''
Analizando el gráfico las asignaturas que muestran mejor separación entre las casas
y que por tanto serían las más relevantes para el modelo de regresión logística son:

1. 'Defense Against the Dark Arts': Muestra una clara separación entre las casas
2. 'Herbology' (Herbología): Particularmente útil para distinguir Hufflepuff
3. 'Potions' (Pociones): Presenta buena separación especialmente para Slytherin
4. 'Charms' (Encantamientos): Ayuda a distinguir Ravenclaw
5. 'Flying' (Vuelo): También muestra patrones de separación útiles, especialmente para Gryffindor

Las asignaturas que muestran mayor solapamiento y por tanto serían menos útiles para el modelo son:

1. Historia de la Magia (History of Magic)
2. Estudios Muggles (Muggle Studies)
3. Runas Antiguas (Ancient Runes)
4. Aritmancia (Arithmancy)
'''