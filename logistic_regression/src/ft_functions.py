import math
import numpy as np

###### FUNCTIONS FOR DESCRIPTIVE STATISTICS ######

def ft_count(data):
    """Calculate the number of non-null observations."""
    return sum(1 for x in data if x is not None and not math.isnan(x))

def ft_mean(data):
    """Calculate the arithmetic mean of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else sum(clean_data) / len(clean_data)

def ft_std(data):
    """Calculate the standard deviation of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 2:
        return float('nan')
    mean = ft_mean(clean_data)
    return math.sqrt(sum((x - mean) ** 2 for x in clean_data) / (len(clean_data) - 1))

def ft_min(data):
    """Find the minimum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else min(clean_data)

def ft_max(data):
    """Find the maximum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else max(clean_data)

def ft_percentile(data, q):
    """Calculate the qth percentile of the data."""
    clean_data = sorted([x for x in data if x is not None and not math.isnan(x)])
    if not clean_data:
        return float('nan')
    if len(clean_data) == 1:
        return clean_data[0]
    position = (len(clean_data) - 1) * q
    floor, ceil = math.floor(position), math.ceil(position)
    if floor == ceil:
        return clean_data[int(position)]
    d0 = clean_data[floor] * (ceil - position)
    d1 = clean_data[ceil] * (position - floor)
    return d0 + d1

def ft_median(data):
    """Calculate the median (50th percentile) of the data."""
    return ft_percentile(data, 0.5)

def ft_iqr(data):
    """Calculate the Interquartile Range (IQR) of the data."""
    q75, q25 = ft_percentile(data, 0.75), ft_percentile(data, 0.25)
    return float('nan') if math.isnan(q75) or math.isnan(q25) else q75 - q25

def ft_skewness(data):
    """Calculate the skewness of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 3:
        return float('nan')
    mean, std = ft_mean(clean_data), ft_std(clean_data)
    if std == 0:
        return float('nan')
    m3 = sum((x - mean) ** 3 for x in clean_data) / len(clean_data)
    return m3 / (std ** 3)

def ft_kurtosis(data):
    """Calculate the kurtosis of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 4:
        return float('nan')
    mean, std = ft_mean(clean_data), ft_std(clean_data)
    if std == 0:
        return float('nan')
    m4 = sum((x - mean) ** 4 for x in clean_data) / len(clean_data)
    return (m4 / (std ** 4)) - 3

def ft_cv(data):
    """Calculate the Coefficient of Variation (CV) of the data."""
    mean, std = ft_mean(data), ft_std(data)
    return float('nan') if mean == 0 or math.isnan(mean) or math.isnan(std) else abs(std / mean)



###### FUNCTIONS FOR LOGISTIC REGRESSION ######

def sigmoid(z):
    """
    Compute the sigmoid function.
    z: array of inputs
    Returns: sigmoid activation for each input
    """
    # Clip values to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross entropy loss.
    y_true: true labels (0 or 1)
    y_pred: predicted probabilities
    Returns: average binary cross entropy loss
    """
    epsilon = 1e-15                                 # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values for numerical stability
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def prepare_one_vs_all(y, positive_class):
    """
    Convert multiclass labels to binary labels for one-vs-all classification.
    y: original multiclass labels
    positive_class: the class to treat as positive (1)
    Returns: binary labels (1 for positive_class, 0 for all others)
    """
    return (y == positive_class).astype(int)


def binary_gradient(X, y_true, y_pred):
    """
    Compute gradient for binary logistic regression.
    X: feature matrix
    y_true: true binary labels
    y_pred: predicted probabilities
    Returns: gradient for weight update
    """
    error = y_pred - y_true
    return np.dot(X.T, error) / X.shape[0]


def predict_one_vs_all(X, weights_dict):
    """
    Make predictions using trained one-vs-all models.
    X: feature matrix
    weights_dict: dictionary of weights for each class
    Returns: predicted class labels
    """
    predictions = {}
    classes = list(weights_dict.keys())
    
    # Calcular probabilidades para cada clase
    for class_name, weights in weights_dict.items():
        predictions[class_name] = sigmoid(np.dot(X, weights))
    
    # Convertir predicciones a array
    pred_array = np.column_stack([predictions[class_name] for class_name in classes])
    
    # Obtener los índices de las probabilidades máximas
    max_indices = np.argmax(pred_array, axis=1)
    
    # Convertir índices a nombres de clases
    return [classes[idx] for idx in max_indices]



###### FUNCTIONS TO MEASURE THE QUALITY OF THE FIT ######

def ft_accuracy_score(y_true, y_pred):
    """
    This function accuracy_score exists in sklearn.
    Calculate accuracy score.
    y_true: array of true labels
    y_pred: array of predicted labels
    Returns: accuracy score between 0 and 1
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def ft_precision_recall_fscore_support(y_true, y_pred, labels=None):
    """
    This function precision_recall_fscore_support exists in sklearn.
    Calculate precision, recall, f1-score and support for each class.
    
    y_true: array of true labels
    y_pred: array of predicted labels
    labels: list of labels to include in the computation
    
    Returns: tuple (precision, recall, f1_score, support)
    Each element is a list with values for each class
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Initialize metrics
    precision = []
    recall = []
    f1_score = []
    support = []
    
    for label in labels:
        # True positives (TP): predicted label correctly
        tp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == label and pred == label)
        
        # False positives (FP): predicted label incorrectly
        fp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true != label and pred == label)
        
        # False negatives (FN): failed to predict label
        fn = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == label and pred != label)
        
        # Support: number of occurrences of label in true labels
        label_support = sum(1 for true in y_true if true == label)
        
        # Calculate metrics
        # Handle division by zero
        label_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        label_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        if label_precision + label_recall > 0:
            label_f1 = 2 * (label_precision * label_recall) / (label_precision + label_recall)
        else:
            label_f1 = 0
        
        # Append metrics for this label
        precision.append(label_precision)
        recall.append(label_recall)
        f1_score.append(label_f1)
        support.append(label_support)
    
    return precision, recall, f1_score, support