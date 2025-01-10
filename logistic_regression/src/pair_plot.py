import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def exploring_feature_relationships():
    print("\nQuestion:\nFrom this visualization, what features are you going to use for your logistic regression?\n")
    print('''    Analyzing the graph, the courses that show better separation between houses
    and therefore would be the most relevant for the logistic regression model are:\n
    \t1. 'Defense Against the Dark Arts': Shows a clear separation between houses
    \t2. 'Herbology': Particularly useful for distinguishing Hufflepuff
    \t3. 'Potions': Presents good separation especially for Slytherin
    \t4. 'Charms': Helps to distinguish Ravenclaw
    \t5. 'Flying': Also shows useful separation patterns, especially for Gryffindor
    
    The courses that show greater overlap and therefore would be less useful for the model are:
    \t- History of Magic
    \t- Muggle Studies
    \t- Ancient Runes
    \t- Arithmancy
    ''')

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