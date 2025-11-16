"""
Model utility functions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs,
):
    """
    model: Model to save
    save_path: Path to save checkpoint
    optimizer: Optimizer state (optional)
    scheduler: Scheduler state (optional)
    epoch: Current epoch (optional)
    metrics: Metrics dictionary (optional)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **kwargs,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(
    model: nn.Module,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    model: Model to load weights into
    load_path: Path to checkpoint
    optimizer: Optimizer to load state into (optional)
    scheduler: Scheduler to load state into (optional)
    device: Device to load on
    r value: Dictionary of loaded information
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    loaded_info = {}
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_info['optimizer'] = True
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loaded_info['scheduler'] = True
    
    if 'epoch' in checkpoint:
        loaded_info['epoch'] = checkpoint['epoch']
    
    if 'metrics' in checkpoint:
        loaded_info['metrics'] = checkpoint['metrics']
    
    print(f"Model loaded from {load_path}")
    return loaded_info


def get_model_summary(model: nn.Module, input_size: tuple = None) -> str:
    summary = []
    summary.append("=" * 80)
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append("=" * 80)
    summary.append(f"Total parameters: {count_parameters(model):,}")
    summary.append(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    summary.append("=" * 80)
    
    return "\n".join(summary)

