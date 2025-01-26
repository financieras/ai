import numpy as np
import pandas as pd
import json
import os
from aux.ft_functions import (
    sigmoid, 
    predict_one_vs_all, 
    ft_accuracy_score, 
    ft_precision_recall_fscore_support
)

class LogisticRegressionPredictor:
    def __init__(self, weights_file):
        """
        Inicializa el predictor cargando los pesos entrenados
        """
        with open(weights_file, 'r') as f:
            weights_dict = json.load(f)
            self.weights = {
                house: np.array(weights)
                for house, weights in weights_dict.items()
            }

    def predict(self, X):
        """
        Predice la casa más probable para cada estudiante usando predict_one_vs_all
        """
        return predict_one_vs_all(X, self.weights)

def evaluate_model(y_true, y_pred):
    """
    Calcula y muestra las métricas de evaluación usando nuestras propias funciones
    """
    accuracy = ft_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = ft_precision_recall_fscore_support(
        y_true, 
        y_pred, 
        labels=['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    )
    
    print("\nMétricas de evaluación:")
    print(f"Accuracy global: {accuracy:.4f}")
    print("\nMétricas por casa:")
    for i, house in enumerate(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']):
        print(f"\n{house}:")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1-score: {f1[i]:.4f}")

def main():
    # Definir rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset_path = os.path.join(parent_dir, 'datasets', 'dataset_normalized.csv')
    weights_path = os.path.join(parent_dir, 'output', 'model_weights.json')
    predictions_path = os.path.join(parent_dir, 'output', 'houses.csv')

    # Cargar datos
    print("Cargando datos...")
    data = pd.read_csv(dataset_path)

    # Preparar características
    feature_columns = ['Best Hand', 'Arithmancy', 'Herbology', 'Defense Against the Dark Arts', 
                      'Divination', 'Muggle Studies', 'Ancient Runes', 
                      'History of Magic', 'Transfiguration', 'Potions', 
                      'Care of Magical Creatures', 'Charms', 'Flying', 'Age']
    
    X = data[feature_columns].values

    # Crear vector de casas reales para evaluación
    y_true = np.where(data['House_Gryffindor'] == 1, 'Gryffindor',
             np.where(data['House_Hufflepuff'] == 1, 'Hufflepuff',
             np.where(data['House_Ravenclaw'] == 1, 'Ravenclaw', 'Slytherin')))

    # Cargar modelo y hacer predicciones
    print("Haciendo predicciones...")
    predictor = LogisticRegressionPredictor(weights_path)
    y_pred = predictor.predict(X)

    # Evaluar el modelo
    evaluate_model(y_true, y_pred)

    # Guardar predicciones en formato requerido
    print("\nGuardando predicciones...")
    predictions_df = pd.DataFrame({
        'Index': range(len(y_pred)),
        'Hogwarts House': y_pred
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predicciones guardadas en {predictions_path}")

if __name__ == "__main__":
    main()