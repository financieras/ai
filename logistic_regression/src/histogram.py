import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_histogram():
    # Read the dataset
    df = pd.read_csv('../datasets/preprocessed_data.csv')
    
    # Get list of course columns (excluding non-course columns)
    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Divination', 'Muggle_Studies',
              'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions',
              'Care_of_Magical_Creatures', 'Charms', 'Flying']
    
    # Create a figure with subplots for each course
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Distribution of Scores by House for Each Course', fontsize=16)
    
    # Create a subplot for each course
    for idx, course in enumerate(courses, 1):
        plt.subplot(4, 3, idx)
        
        # Create histogram for current course
        sns.histplot(data=df, x=course, hue='Hogwarts_House', 
                    bins=20, alpha=0.5, multiple="layer")
        
        plt.title(course.replace('_', ' '))
        plt.xlabel('Score')
        plt.ylabel('Count')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Only show legend for the first subplot to avoid redundancy
        if idx != 1:
            plt.legend([],[], frameon=False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('../output/histogram.png')
    plt.close()

    print("\nThe histogram graph has been saved in the 'output' folder.")

if __name__ == "__main__":
    generate_histogram()


'''
Respuesta a la pregunta:

> "Which Hogwarts course has a homogeneous score distribution between all four houses?"

Podemos descartar los histogramas bimodales y quedarnos con las siguientes tres opciones:

1. **`Arithmancy`**
- Muestra una distribución bastante superpuesta
- Sin embargo, se puede ver que `Hufflepuff` (rojo) tiene una ligera tendencia hacia puntuaciones más altas
- `Ravenclaw` (azul) tiene una concentración ligeramente menor en el pico central

2. **`Potions`**
- Aunque hay superposición, las distribuciones están bastante dispersas
- Se pueden ver diferencias claras entre casas
- `Slytherin` (verde) tiende a tener puntuaciones más altas
-Las distribuciones son más anchas y menos uniformes entre casas

3. **`Care of Magical Creatures`**
- Muestra la superposición más uniforme de los tres
- Los picos de todas las casas están prácticamente en el mismo punto
- Las formas de las distribuciones son muy similares para todas las casas
- La dispersión (ancho de la distribución) es muy similar para todas las casas
- Es difícil distinguir diferencias significativas entre casas


**Conclusión**
- La distribución más uniforme es **`Care of Magical Creatures`**.
- Este curso muestra la distribución más equilibrada y similar entre todas las casas, con patrones de puntuación muy parecidos independientemente de la casa a la que pertenezcan los estudiantes.
'''