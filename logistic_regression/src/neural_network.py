import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, random_state=None):
        """
        Initialize a neural network for multinomial logistic regression.
        
        Parameters:
        layer_sizes (list): List containing the number of neurons in each layer
                          [input_size, hidden1_size, hidden2_size, output_size]
        random_state (int): Seed for random number generation (optional)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Xavier/Glorot initialization for weights
        for i in range(self.num_layers - 1):
            # Calculate limit for Xavier/Glorot uniform initialization
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            
            # Initialize weights with Xavier/Glorot uniform initialization
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
            
            # Initialize biases with zeros
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def relu(self, Z):
        """
        ReLU activation function.
        """
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """
        Derivative of ReLU activation function.
        """
        return np.where(Z > 0, 1, 0)
    
    def softmax(self, Z):
        """
        Softmax activation function for output layer.
        Numerically stable implementation.
        """
        # Shift values to avoid overflow
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward propagation step.
        
        Parameters:
        X (np.array): Input data of shape (n_samples, n_features)
        
        Returns:
        tuple: List of activations and outputs for each layer
        """
        # Store all activations and layer outputs
        activations = [X]  # List to store activations of each layer
        Z_values = []      # List to store Z values (pre-activation)
        
        # Forward propagate through hidden layers (using ReLU)
        for i in range(self.num_layers - 2):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            activations.append(self.relu(Z))
        
        # Output layer (using Softmax)
        Z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        Z_values.append(Z)
        activations.append(self.softmax(Z))
        
        return activations, Z_values