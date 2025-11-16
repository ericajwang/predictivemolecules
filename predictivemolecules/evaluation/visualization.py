import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
from .metrics import calculate_rmse, calculate_r2, calculate_pearson


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Predictions vs True Values',
    figsize: tuple = (8, 6),
):
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    pearson, _ = calculate_pearson(y_true, y_pred)
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    
    textstr = f'RMSE: {rmse:.3f}\nR²: {r2:.3f}\nPearson: {pearson:.3f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Residual Plot',
    figsize: tuple = (10, 6),
):
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predictions', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    textstr = f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}'
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_learning_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5),
):
    num_plots = 1
    if train_metrics or val_metrics:
        num_plots += 1
    
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Learning Curves - Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if num_plots > 1 and (train_metrics or val_metrics):
        if train_metrics:
            for metric_name, values in train_metrics.items():
                if len(values) == len(epochs):
                    axes[1].plot(epochs, values, label=f'Train {metric_name}', linewidth=2)
        
        if val_metrics:
            for metric_name, values in val_metrics.items():
                if len(values) == len(epochs):
                    axes[1].plot(epochs, values, '--', label=f'Val {metric_name}', linewidth=2)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Metric Value', fontsize=12)
        axes[1].set_title('Learning Curves - Metrics', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_correlation_matrix(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    
    all_metrics = sorted(list(all_metrics))
    
    matrix_data = []
    model_names = []
    for model_name, model_metrics in metrics_dict.items():
        row = [model_metrics.get(metric, np.nan) for metric in all_metrics]
        matrix_data.append(row)
        model_names.append(model_name)
    
    matrix = np.array(matrix_data)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        xticklabels=all_metrics,
        yticklabels=model_names,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Metric Value'},
    )
    plt.title('Model Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

