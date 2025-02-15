import math
import numpy as np
import pandas as pd


###### FUNCTIONS FOR DEEP NEURAL NETWORK ######

def relu(Z):
    """
    Función de activación ReLU
    """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Derivada de la función ReLU
    """
    return np.where(Z > 0, 1, 0)

def sigmoid(Z):
    """
    Función de activación Sigmoid
    """
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    """
    Derivada de la función Sigmoid
    """
    s = sigmoid(Z)
    return s * (1 - s)

def initialize_parameters(layer_dims):
    """
    Inicializa los parámetros de la red neuronal con la inicialización He
    """
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    return parameters