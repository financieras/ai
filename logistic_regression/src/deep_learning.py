import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=5, hidden1_size=8, hidden2_size=6, output_size=4, learning_rate=0.01):
        """
        Initialize neural network with specified architecture
        
        Parameters:
        input_size (int): Number of input features
        hidden1_size (int): Number of neurons in first hidden layer
        hidden2_size (int): Number of neurons in second hidden layer
        output_size (int): Number of output classes
        learning_rate (float): Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0/input_size)
        self.bias1 = np.zeros((1, hidden1_size))
        
        self.weights2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0/hidden1_size)
        self.bias2 = np.zeros((1, hidden2_size))
        
        self.weights3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0/hidden2_size)
        self.bias3 = np.zeros((1, output_size))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        
        Parameters:
        X (np.array): Input data of shape (n_samples, input_size)
        
        Returns:
        tuple: Activation values for each layer
        """
        # First hidden layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        
        # Second hidden layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        
        # Output layer
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.softmax(self.z3)
        
        return self.a3
    
    def cross_entropy_loss(self, y_true, y_pred):
        """Calculate cross entropy loss"""
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values for numerical stability
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, X, y_true, y_pred):
        """
        Backward propagation
        
        Parameters:
        X (np.array): Input data
        y_true (np.array): True labels
        y_pred (np.array): Predicted probabilities
        """
        batch_size = X.shape[0]
        
        # Output layer error
        delta3 = y_pred - y_true
        
        # Second hidden layer error
        delta2 = np.dot(delta3, self.weights3.T) * self.relu_derivative(self.z2)
        
        # First hidden layer error
        delta1 = np.dot(delta2, self.weights2.T) * self.relu_derivative(self.z1)
        
        # Update weights and biases
        self.weights3 -= self.learning_rate * np.dot(self.a2.T, delta3) / batch_size
        self.bias3 -= self.learning_rate * np.mean(delta3, axis=0, keepdims=True)
        
        self.weights2 -= self.learning_rate * np.dot(self.a1.T, delta2) / batch_size
        self.bias2 -= self.learning_rate * np.mean(delta2, axis=0, keepdims=True)
        
        self.weights1 -= self.learning_rate * np.dot(X.T, delta1) / batch_size
        self.bias1 -= self.learning_rate * np.mean(delta1, axis=0, keepdims=True)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, early_stopping_patience=5):
        """
        Train the neural network
        
        Parameters:
        X_train (np.array): Training data
        y_train (np.array): Training labels
        X_val (np.array): Validation data
        y_val (np.array): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Size of training batches
        early_stopping_patience (int): Number of epochs to wait before early stopping
        
        Returns:
        dict: Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, y_pred)
                
                # Calculate metrics
                batch_loss = self.cross_entropy_loss(y_batch, y_pred)
                batch_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                
                epoch_loss += batch_loss * len(X_batch)
                epoch_acc += batch_acc * len(X_batch)
            
            # Calculate epoch metrics
            epoch_loss /= n_samples
            epoch_acc /= n_samples
            
            # Validation metrics
            val_pred = self.forward(X_val)
            val_loss = self.cross_entropy_loss(y_val, val_pred)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            
            # Store metrics
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(epoch_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
        
        return history
    
    def predict(self, X):
        """Make predictions for input data"""
        return self.forward(X)

if __name__ == "__main__":
    # Example usage (assuming data is already prepared)
    # model = NeuralNetwork()
    # history = model.train(X_train, y_train, X_val, y_val)
    pass