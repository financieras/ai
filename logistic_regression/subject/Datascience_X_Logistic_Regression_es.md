# Regresión logística en ciencia de datos
## Harry Potter y un Científico de Datos
- Resumen: ¡Escribe un clasificador y salva Hogwarts!
- Versión: 2

## Contenidos

1. Prólogo
2. Introducción
3. Objetivos
4. Instrucciones generales
5. Parte obligatoria
   - 5.1. Análisis de datos
   - 5.2. Visualización de datos
     - 5.2.1. Histograma
     - 5.2.2. Gráfico de dispersión
     - 5.2.3. Gráfico de pares
   - 5.3 Regresión Logística
6. Parte Bonus
7. Entrega y evaluación entre pares
8. Anexos
   - 8.1. Matemáticas
   - 8.2. Ejemplos de visualización de datos

## Capítulo I Prólogo

Esto es lo que Wikipedia dice sobre Yann Le Cun, uno de los padres fundadores de la IA:

Yann Le Cun nació cerca de París, Francia, en 1960. Es un investigador en inteligencia artificial y visión por computadora (robótica). Es considerado uno de los inventores del aprendizaje profundo. Obtuvo una licenciatura de la ESIEE Paris en 1983, se graduó de la Universidad Pierre y Marie Curie y recibió un doctorado en 1987, durante el cual propuso una forma temprana del algoritmo de aprendizaje de retropropagación que se usa comúnmente en los algoritmos de optimización de descenso de gradiente para ajustar el peso de las neuronas calculando el gradiente de la función de pérdida. Fue investigador postdoctoral asociado en el laboratorio de Geoffrey Hinton en la Universidad de Toronto de 1987 a 1988.

Desde la década de 1980, Yann Le Cun ha estado trabajando en aprendizaje automático, visión por computadora y aprendizaje profundo: la capacidad de una computadora para reconocer representaciones (imágenes, textos, videos, sonidos) exponiéndolas repetidamente a las muestras de entrenamiento.

En 1987, Yann Le Cun se unió a la Universidad de Toronto y en 1988 a las instalaciones de investigación de AT&T, donde desarrolló los métodos de aprendizaje supervisado. Yann Le Cun es también uno de los principales creadores de la tecnología de compresión de imágenes DjVu (junto con Léon Bottou y Patrick Haffner).

Yann Le Cun es profesor en la Universidad de Nueva York, donde ha creado el Centro de Ciencia de Datos. Trabaja en particular en el desarrollo tecnológico de automóviles autónomos. El 9 de diciembre de 2013, Yann Le Cun fue invitado por Mark Zuckerberg a unirse a Facebook para diseñar y dirigir el laboratorio de inteligencia artificial FAIR ("Facebook Artificial Intelligence Research") en Nueva York, Menlo Park y desde 2015 en París, para trabajar en el reconocimiento de imágenes. Anteriormente había rechazado una propuesta similar de Google.

En 2016, fue profesor visitante de ciencias de la computación en la "Chaire Annuelle Informatique et Sciences Numériques" en el Collège de France en París.

## Capítulo II Introducción

¡Oh no! Desde su creación, la famosa escuela de magos, Hogwarts, nunca había conocido tal ofensa. Las fuerzas del mal han embrujado el Sombrero Seleccionador. Ya no responde y es incapaz de cumplir su papel de clasificar a los estudiantes en las casas. El nuevo año académico se acerca. Afortunadamente, la Profesora McGonagall pudo tomar medidas en una situación tan estresante, ya que es imposible que Hogwarts no reciba nuevos estudiantes...

Decidió llamarte a ti, un "científico de datos" muggle que es capaz de crear milagros con la herramienta que todos los muggles saben usar: una "computadora". A pesar de la reticencia intrínseca de muchos magos, el director de la escuela te recibe en su oficina para explicarte la situación. Estás aquí porque su informante descubrió que eres capaz de recrear un Sombrero Seleccionador mágico usando tus herramientas muggle.

Le explicas que para que tus herramientas "muggle" funcionen, necesitas datos de los estudiantes. Dudando, la Profesora McGonagall te entrega un polvoriento libro de hechizos. Afortunadamente para ti, un simple "¡Digitalis!" y el libro se convirtió en una memoria USB.

## Capítulo III Objetivos

En este proyecto DataScience x Regresión Logística, continuarás tu exploración del Aprendizaje Automático descubriendo diferentes herramientas. El uso del término DataScience en el título será claramente considerado por algunos como abusivo. Eso es cierto. No pretendemos darte todas las bases de la Ciencia de Datos en este tema. El tema es vasto. Solo veremos aquí algunas bases que nos parecieron útiles para la exploración de datos antes de enviarlos al algoritmo de aprendizaje automático.

Implementarás un **modelo de clasificación lineal**, como continuación del tema de regresión lineal: una **regresión logística**. También te alentamos mucho a crear un conjunto de **herramientas de aprendizaje automático** mientras avanzas en la rama.

Resumiendo:
  - Aprenderás a leer un conjunto de datos, visualizarlo de diferentes maneras, seleccionar y limpiar información innecesaria de tus datos.
  - Entrenarás una regresión logística que resolverá un problema de clasificación.

## Capítulo IV Instrucciones generales

Puedes usar el lenguaje que quieras. Sin embargo, te recomendamos que elijas un lenguaje con una biblioteca que facilite el trazado y el cálculo de propiedades estadísticas de un conjunto de datos. Cualquier función que haga todo el trabajo pesado por ti (por ejemplo, usar la función `describe` de la biblioteca `Pandas`) se considerará trampa.

## Capítulo V Parte obligatoria

Se recomienda encarecidamente realizar los pasos en el siguiente orden.

### V.1 Análisis de datos

Veremos algunos pasos básicos de exploración de datos. Por supuesto, estas no son las únicas técnicas disponibles ni el único paso a seguir. Cada conjunto de datos y problema debe abordarse de una manera única. Seguramente encontrarás otras formas de analizar tus datos en el futuro.

En primer lugar, echa un vistazo a los datos disponibles. Mira en qué formato se presentan, si hay varios tipos de datos, los diferentes rangos, etc. Es importante hacerse una idea de tu materia prima antes de comenzar. Cuanto más trabajes con datos, más desarrollarás una intuición sobre cómo podrás usarlos.

En esta parte, la Profesora McGonagall te pide que produzcas un programa llamado `describe.[extension]`. Este programa tomará un conjunto de datos como parámetro. Todo lo que tiene que hacer es mostrar información para todas las características numéricas como en el ejemplo:

```bash
$> describe.[extension] dataset_train.csv
        Feature_01     Feature_02      Feature_03      Feature_04
Count   149.000000     149.000000      149.000000      149.000000
Mean      5.848322       3.051007        3.774497        1.205369
Std       5.906338       3.081445        4.162021        1.424286
Min       4.300000       2.000000        1.000000        0.100000
25%       5.100000       2.800000        1.600000        0.300000
50%       5.800000       3.000000        4.400000        1.300000
75%       6.400000       3.300000        5.100000        1.800000
Max       7.900000       4.400000        6.900000        2.500000
```

> Está prohibido usar cualquier función que haga el trabajo por ti como: `count`, `mean`, `std`, `min`, `max`, `percentile`, etc... sin importar el lenguaje que uses. Por supuesto, también está prohibido usar la biblioteca `describe` o cualquier función que se parezca (más o menos) a ella de otra biblioteca.

### V.2 Visualización de datos

La visualización de datos es una herramienta poderosa para un científico de datos. Te permite obtener ideas y desarrollar una intuición de cómo son tus datos. Visualizar tus datos también te permite detectar defectos o anomalías.

En esta sección, se te pide que crees un conjunto de scripts, cada uno utilizando un método de visualización particular para responder una pregunta. No hay necesariamente una sola respuesta a la pregunta.

#### V.2.1 Histograma

Haz un script llamado `histogram.[extension]` que muestre un histograma respondiendo a la siguiente pregunta:
¿Qué curso de Hogwarts tiene una distribución de puntajes homogénea entre las cuatro casas?

#### V.2.2 Gráfico de dispersión

Haz un script llamado `scatter_plot.[extension]` que muestre un gráfico de dispersión respondiendo a la siguiente pregunta:
¿Cuáles son las dos características que son similares?

#### V.2.3 Gráfico de pares

Haz un script llamado `pair_plot.[extension]` que muestre un gráfico de pares o una matriz de gráficos de dispersión (según la biblioteca que estés usando).
A partir de esta visualización, ¿qué características vas a usar para tu regresión logística?

### V.3 Regresión Logística

Llegas a la última parte: codifica tu Sombrero Mágico. Para hacer esto, tienes que realizar un multi-clasificador utilizando una regresión logística uno contra todos.

Tendrás que hacer dos programas:

• El primero entrenará tus modelos, se llama `logreg_train.[extension]`. Toma como parámetro `dataset_train.csv`. Para la parte obligatoria, debes usar la técnica de descenso de gradiente para minimizar el error. El programa genera un archivo que contiene los pesos que se utilizarán para la predicción.

• El segundo debe llamarse `logreg_predict.[extension]`. Toma como parámetro `dataset_test.csv` y un archivo que contiene los pesos entrenados por el programa anterior.

Para evaluar el rendimiento de tu clasificador, este segundo programa tendrá que generar un archivo de predicción `houses.csv` formateado exactamente de la siguiente manera:

```bash
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Hufflepuff
4,Slytherin
5,Ravenclaw
6,Hufflepuff
[...]
```

## Capítulo VI Parte Bonus

Es posible hacer muchos bonos interesantes para este tema. Aquí hay algunas sugerencias:

   1. Agregar más campos para `describe.[extension]`
   2. Implementar un *descenso de gradiente estocástico*
   3. Implementar otros algoritmos de optimización (GD por lotes/GD mini-lotes/como lo llames)

> La parte de bonificación solo se evaluará si la parte obligatoria es PERFECTA. Perfecto significa que la parte obligatoria se ha realizado íntegramente y funciona sin fallos. Si no has pasado TODOS los requisitos obligatorios, tu parte de bonificación no se evaluará en absoluto.

## Capítulo VII Entrega y evaluación entre pares

Entrega tu tarea en tu repositorio Git como de costumbre. Solo el trabajo dentro de tu repositorio será evaluado durante la defensa. No dudes en verificar dos veces los nombres de tus carpetas y archivos para asegurarte de que sean correctos.

Durante la corrección, serás evaluado en tu entrega (sin funciones que hagan todo el trabajo pesado por ti), así como en tu capacidad para presentar, explicar y justificar tus elecciones.

Tu clasificador será evaluado con los datos presentes en `dataset_test.csv`. Tus respuestas serán evaluadas utilizando la [puntuación de precisión](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) de la biblioteca `Scikit-Learn`.

La Profesora McGonagall está de acuerdo en que tu algoritmo es comparable al Sombrero Seleccionador solo si tiene una precisión mínima del **98%**.

También será importante poder explicar el funcionamiento de los algoritmos de aprendizaje automático utilizados.

## Capítulo VIII Anexos

### VIII.1 Matemáticas

La regresión logística funciona casi como la regresión lineal. Aquí hay una función de costo (pérdida):

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(h_\theta(x_i)) + (1-y_i) \log(1-h_\theta(x_i))]$$

Donde $h_\theta(x)$ se define de la siguiente manera:

$$h_\theta(x) = g(\theta^T x)$$

Con:

$$g(z) = \frac{1}{1 + e^{-z}}$$

La función de pérdida nos da la siguiente derivada parcial:

$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x_i) - y_i)x_i^j$$

### VIII.2 Ejemplos de visualización de datos

Aquí hay algunos ejemplos de visualización de datos:

• Histograma

<img src="https://github.com/financieras/ai/blob/main/logistic_regression/subject/histogram.png?raw=1" alt="histogram" width="400"/>

• Gráfico de dispersión

<img src="https://github.com/financieras/ai/blob/main/logistic_regression/subject/scratter_plot.png?raw=1" alt="scratter_plot" width="400"/>

• Gráfico de pares

<img src="https://github.com/financieras/ai/blob/main/logistic_regression/subject/pair_plot.png?raw=1" alt="pair_plot" width="600"/>
