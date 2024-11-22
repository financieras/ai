import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def exploring_feature_relationships():
    # Cargar los datos
    df = pd.read_csv('../datasets/preprocessed_data.csv')

    # Select numerical columns of type float64
    #features = df.select_dtypes(include=['float64'])
    features = ['Arithmancy', 'Astronomy', 'Herbology', 'Divination', 'Muggle_Studies', 'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions', 'Care_of_Magical_Creatures', 'Charms', 'Flying']
  
    # Crear el pair plot
    sns.pairplot(df, hue='Hogwarts_House', vars=features)

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

1. Astronomía: Muestra una clara separación entre las casas
2. Herbología: Particularmente útil para distinguir Hufflepuff
3. Pociones: Presenta buena separación especialmente para Slytherin
4. Encantamientos (Charms): Ayuda a distinguir Ravenclaw
5. Flying (Vuelo): También muestra patrones de separación útiles, especialmente para Gryffindor

Las asignaturas que muestran mayor solapamiento y por tanto serían menos útiles para el modelo son:

1. Historia de la Magia
2. Estudios Muggles (Muggle Studies)
3. Runas Antiguas (Ancient Runes)
4. Aritmancia (Arithmancy)
'''