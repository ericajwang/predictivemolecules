import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets, reduction=self.reduction)


class MAELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(MAELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(predictions, targets, reduction=self.reduction)


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        delta: Threshold for transition between L1 and L2 loss
        reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(predictions, targets, delta=self.delta, reduction=self.reduction)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9], reduction: str = 'mean'):
        """
        quantiles: List of quantiles to predict
        reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.shape[-1] != len(self.quantiles):
            raise ValueError(
                f"Predictions must have {len(self.quantiles)}, cur have {predictions.shape[-1]}"
            )
        
        losses = []
        targets = targets.unsqueeze(-1)  # [batch_size, 1]
        
        for i, quantile in enumerate(self.quantiles):
            error = targets - predictions[:, i:i+1]
            loss = torch.max(
                quantile * error,
                (quantile - 1) * error
            )
            losses.append(loss)
        
        total_loss = torch.stack(losses).sum(dim=0)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        loss_types: list = ['mse', 'mae'],
        weights: Optional[list] = None,
    ):
        """
        loss_types: List of loss types ('mse', 'mae', 'huber')
        weights: Weights for each loss (if None, equal weights)
        """
        super(CombinedLoss, self).__init__()
        
        self.loss_functions = nn.ModuleList()
        for loss_type in loss_types:
            if loss_type == 'mse':
                self.loss_functions.append(MSELoss())
            elif loss_type == 'mae':
                self.loss_functions.append(MAELoss())
            elif loss_type == 'huber':
                self.loss_functions.append(HuberLoss())
        
        if weights is None:
            self.weights = [1.0 / len(loss_types)] * len(loss_types)
        else:
            assert len(weights) == len(loss_types)
            self.weights = weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.loss_functions, self.weights):
            total_loss += weight * loss_fn(predictions, targets)
        
        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = (predictions - targets) ** 2
        mse_mean = mse.mean()
        
        weights = (mse / (mse_mean + 1e-8)) ** self.gamma
        
        focal_loss = self.alpha * weights * mse
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

