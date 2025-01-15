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
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
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
    for class_name, weights in weights_dict.items():
        predictions[class_name] = sigmoid(np.dot(X, weights))
    
    # Convert predictions dictionary to array
    pred_array = np.column_stack([predictions[class_name] for class_name in weights_dict.keys()])
    
    # Return the class with highest probability
    return list(weights_dict.keys())[np.argmax(pred_array, axis=1)]