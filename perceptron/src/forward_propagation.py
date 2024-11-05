# perceptron/src/forward_propagation.py
# versión con una única capa oculta

import numpy as np
import pandas as pd

def forward_propagation(X, weights, biases):
    """
    Realiza la propagación hacia adelante (forward propagation) en la red neuronal.
    
    Parámetros:
    X (numpy.ndarray): Matriz de características de entrada (muestras x características)
    weights (list): Lista de matrices de pesos para cada capa
    biases (list): Lista de vectores de sesgos para cada capa
    
    Retorna:
    numpy.ndarray: Vector de salidas de la red neuronal
    """
    
    # Número de capas en la red
    num_layers = len(weights)
    
    # Inicializar la activación de la capa de entrada
    activation = X
    
    # Realizar la propagación hacia adelante capa por capa
    for l in range(num_layers):
        # Calcular la entrada ponderada (z)
        z = np.dot(activation, weights[l].T) + biases[l]
        
        # Aplicar la función de activación (a)
        activation = sigmoid(z)
    
    # La salida de la red neuronal es la activación de la última capa
    output = activation
    
    return output

def sigmoid(z):
    """
    Aplica la función de activación sigmoide a un valor o array.
    
    Parámetro:
    z (float o numpy.ndarray): Valor o array de valores a los que aplicar la función sigmoide
    
    Retorna:
    float o numpy.ndarray: Valor o array con los resultados de aplicar la función sigmoide
    """
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    # Cargar los datos preprocesados
    data = pd.read_csv('data/preprocessed_data.csv')
    
    # Separar las características (X) y el objetivo (y)
    X = data.iloc[:, 2:].values  # Características (features)
    y = data['diagnosis'].values  # Objetivo (diagnóstico)
    
    # Definir la estructura de la red neuronal
    num_inputs = X.shape[1]
    num_hidden = 10
    num_outputs = 1

    # Inicializar aleatoriamente los pesos y sesgos
    W1 = np.random.randn(num_hidden, num_inputs)
    b1 = np.random.randn(1, num_hidden)  # (1, 10)
    W2 = np.random.randn(num_outputs, num_hidden)
    b2 = np.random.randn(1, num_outputs)  # (1, 1)

    weights = [W1, W2]
    biases = [b1, b2]

    # Realizar la propagación hacia adelante
    predictions = forward_propagation(X, weights, biases)
    print("Predicciones:", predictions)