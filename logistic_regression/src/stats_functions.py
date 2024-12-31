import math
from typing import List, Union, Optional
import numpy as np

def ft_count(data: List[Union[float, int]]) -> int:
    """Calculate the number of non-null observations."""
    return sum(1 for x in data if x is not None and not math.isnan(x))

def ft_mean(data: List[Union[float, int]]) -> float:
    """Calculate the arithmetic mean of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if not clean_data:
        return float('nan')
    return sum(clean_data) / len(clean_data)

def ft_std(data: List[Union[float, int]]) -> float:
    """Calculate the standard deviation of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 2:
        return float('nan')
    
    mean = ft_mean(clean_data)
    squared_diff_sum = sum((x - mean) ** 2 for x in clean_data)
    return math.sqrt(squared_diff_sum / (len(clean_data) - 1))

def ft_min(data: List[Union[float, int]]) -> float:
    """Find the minimum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if not clean_data:
        return float('nan')
    return min(clean_data)

def ft_max(data: List[Union[float, int]]) -> float:
    """Find the maximum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if not clean_data:
        return float('nan')
    return max(clean_data)

def ft_percentile(data: List[Union[float, int]], q: float) -> float:
    """Calculate the qth percentile of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if not clean_data:
        return float('nan')
    
    sorted_data = sorted(clean_data)
    n = len(sorted_data)
    
    if n == 1:
        return sorted_data[0]
    
    position = (n - 1) * q
    floor = math.floor(position)
    ceil = math.ceil(position)
    
    if floor == ceil:
        return sorted_data[int(position)]
    
    d0 = sorted_data[int(floor)] * (ceil - position)
    d1 = sorted_data[int(ceil)] * (position - floor)
    return d0 + d1

def ft_median(data: List[Union[float, int]]) -> float:
    """Calculate the median (50th percentile) of the data."""
    return ft_percentile(data, 0.5)

def ft_iqr(data: List[Union[float, int]]) -> float:
    """Calculate the Interquartile Range (IQR) of the data."""
    q75 = ft_percentile(data, 0.75)
    q25 = ft_percentile(data, 0.25)
    
    if math.isnan(q75) or math.isnan(q25):
        return float('nan')
    
    return q75 - q25

def ft_skewness(data: List[Union[float, int]]) -> float:
    """Calculate the skewness of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 3:
        return float('nan')
    
    n = len(clean_data)
    mean = ft_mean(clean_data)
    std = ft_std(clean_data)
    
    if std == 0:
        return float('nan')
    
    m3 = sum((x - mean) ** 3 for x in clean_data) / n
    return m3 / (std ** 3)

def ft_kurtosis(data: List[Union[float, int]]) -> float:
    """Calculate the kurtosis of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 4:
        return float('nan')
    
    n = len(clean_data)
    mean = ft_mean(clean_data)
    std = ft_std(clean_data)
    
    if std == 0:
        return float('nan')
    
    m4 = sum((x - mean) ** 4 for x in clean_data) / n
    return (m4 / (std ** 4)) - 3

def ft_cv(data: List[Union[float, int]]) -> float:
    """Calculate the Coefficient of Variation (CV) of the data."""
    mean = ft_mean(data)
    std = ft_std(data)
    
    if mean == 0 or math.isnan(mean) or math.isnan(std):
        return float('nan')
    
    return abs(std / mean)