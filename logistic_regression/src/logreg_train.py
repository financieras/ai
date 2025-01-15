import numpy as np
import pandas as pd
import os
import json
from stats_functions import (
    sigmoid, 
    binary_cross_entropy, 
    prepare_one_vs_all, 
    binary_gradient
)

class LogisticRegressionTrainer:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = {}
        
    def train_one_vs_all(self, X, y, house):
        """Train a single binary classifier for one house vs all others"""
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        y_binary = prepare_one_vs_all(y, house)
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = sigmoid(np.dot(X, weights))
            
            # Calculate loss
            current_loss = binary_cross_entropy(y_binary, y_pred)
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                break
                
            # Compute gradients and update weights
            gradients = binary_gradient(X, y_binary, y_pred)
            weights -= self.learning_rate * gradients
            
            prev_loss = current_loss
            
        return weights
    
    def train(self, X, houses):
        """Train all binary classifiers"""
        unique_houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        
        for house in unique_houses:
            print(f"Training classifier for {house}...")
            self.weights[house] = self.train_one_vs_all(X, houses, house)
            
        return self.weights

def main():
    # Definir rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset_path = os.path.join(parent_dir, 'datasets', 'dataset_normalized.csv')
    
    # Crear directorio output si no existe
    output_dir = os.path.join(parent_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, 'model_weights.json')

    # Cargar y preparar datos
    print("Loading data...")
    data = pd.read_csv(dataset_path)
    
    # Separar características y etiquetas
    # Excluimos las columnas de casas
    feature_columns = ['Best Hand', 'Arithmancy', 'Herbology', 'Defense Against the Dark Arts', 
                      'Divination', 'Muggle Studies', 'Ancient Runes', 
                      'History of Magic', 'Transfiguration', 'Potions', 
                      'Care of Magical Creatures', 'Charms', 'Flying', 'Age']
    
    X = data[feature_columns].values
    
    # Crear vector de casas
    houses = np.where(data['House_Gryffindor'] == 1, 'Gryffindor',
             np.where(data['House_Hufflepuff'] == 1, 'Hufflepuff',
             np.where(data['House_Ravenclaw'] == 1, 'Ravenclaw', 'Slytherin')))
    
    # Entrenar modelo
    print("Training model...")
    trainer = LogisticRegressionTrainer(
        learning_rate=0.01,
        max_iterations=1000,
        tolerance=1e-4
    )
    
    weights = trainer.train(X, houses)
    
    # Convertir los pesos a un formato serializable
    weights_dict = {
        house: weights[house].tolist() for house in weights
    }
    
    # Guardar pesos en formato JSON
    print("Saving weights...")
    with open(weights_path, 'w') as f:
        json.dump(weights_dict, f, indent=4)
    print(f"Weights saved to {weights_path}")

if __name__ == "__main__":
    main()