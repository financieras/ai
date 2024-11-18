import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ai/logistic_regression/datasets/preprocessed_data.csv", index_col=0)
# index_col=0 indica que la primera columna del archivo CSV contiene el índice del DataFrame

df.info()


df.head()   # muestra las cinco primeras filas


# Student Distribution across Hogwarts Houses

import matplotlib.pyplot as plt

# Assuming your data is in a pandas DataFrame called 'df'
house_counts = df['Hogwarts_House'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(house_counts, labels=house_counts.index, autopct='%1.1f%%', startangle=90)
_ = plt.title('Student Distribution across Hogwarts Houses')


# Hogwarts_House
from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('Hogwarts_House').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)


# Herbology

from matplotlib import pyplot as plt
df['Herbology'].plot(kind='hist', bins=20, title='Herbology')
plt.gca().spines[['top', 'right',]].set_visible(False)


# Astronomy

from matplotlib import pyplot as plt
df['Astronomy'].plot(kind='hist', bins=20, title='Astronomy')
plt.gca().spines[['top', 'right',]].set_visible(False)


# Arithmancy

from matplotlib import pyplot as plt
df['Arithmancy'].plot(kind='hist', bins=20, title='Arithmancy')
plt.gca().spines[['top', 'right',]].set_visible(False)


# Astronomy vs Herbology
# The most important graphic for the Sorting Hat
from matplotlib import pyplot as plt
df.plot(kind='scatter', x='Astronomy', y='Herbology', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# Bar diagram to Hogwarts House
import seaborn as sns
sns.countplot(df['Hogwarts_House'])