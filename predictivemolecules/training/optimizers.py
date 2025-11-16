import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from typing import Optional, Dict, Any


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adam',
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw', 'rmsprop')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)        
    """
    params = model.parameters()
    
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"wrong argument: {optimizer_type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'reduce_on_plateau',
    **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler
    """
    if scheduler_type is None or scheduler_type == 'none':
        return None
    
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'reduce_on_plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 10)
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=True,
        )
    
    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 50)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    
    elif scheduler_type == 'cosine_restart':
        T_0 = kwargs.get('T_0', 10)
        T_mult = kwargs.get('T_mult', 2)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
        )
    
    else:
        raise ValueError(f"wrong argument: {scheduler_type}")

