import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr
from typing import Dict, List, Optional
import torch


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)


def calculate_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    correlation, p_value = pearsonr(y_true, y_pred)
    return correlation, p_value


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_pearson: bool = True,
) -> Dict[str, float]:
    metrics = {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
    }
    
    if include_pearson:
        correlation, p_value = calculate_pearson(y_true, y_pred)
        metrics['pearson_correlation'] = correlation
        metrics['pearson_p_value'] = p_value
    
    metrics['mean_error'] = np.mean(y_pred - y_true)
    metrics['std_error'] = np.std(y_pred - y_true)
    
    errors = np.abs(y_pred - y_true)
    metrics['within_1_log'] = np.mean(errors < 1.0) * 100
    metrics['within_2_log'] = np.mean(errors < 2.0) * 100
    
    return metrics


class MetricTracker:
    # tracks metric times during training
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        if metric_name in self.metrics_history and self.metrics_history[metric_name]:
            return self.metrics_history[metric_name][-1]
        return None
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Optional[float]:
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return None
        
        values = self.metrics_history[metric_name]
        if mode == 'min':
            return min(values)
        else:
            return max(values)
    
    def get_history(self, metric_name: str) -> List[float]:
        return self.metrics_history.get(metric_name, [])
    
    def reset(self):
        self.metrics_history = {}

