import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
import numpy as np


class EnsembleModel(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        strategy: str = 'weighted_average',
        weights: Optional[List[float]] = None,
        learnable_weights: bool = False,
    ):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.num_models = len(models)
        
        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models
        
        if learnable_weights:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        if strategy == 'stacking':
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )
    
    def forward(
        self,
        inputs: Union[Dict, torch.Tensor],
        return_individual: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        individual_predictions = []
        
        for model in self.models:
            model.eval() if not self.training else model.train()
            
            if isinstance(inputs, dict):
                pred = model(inputs['graph'])
            else:
                pred = model(inputs)
            
            individual_predictions.append(pred)
        
        stacked = torch.stack(individual_predictions, dim=1)
        
        if self.strategy == 'weighted_average':
            normalized_weights = torch.softmax(self.weights, dim=0)
            ensemble_pred = (stacked * normalized_weights.unsqueeze(0)).sum(dim=1)
        
        elif self.strategy == 'stacking':
            ensemble_pred = self.meta_learner(stacked)
            ensemble_pred = ensemble_pred.squeeze(-1)
        
        elif self.strategy == 'voting':
            ensemble_pred = stacked.mean(dim=1)
        
        else:
            ensemble_pred = stacked.mean(dim=1)
        
        if return_individual:
            return ensemble_pred, individual_predictions
        return ensemble_pred
    
    def get_model_weights(self) -> torch.Tensor:
        if self.strategy == 'weighted_average':
            return torch.softmax(self.weights, dim=0)
        return self.weights


class AdaptiveEnsemble(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        feature_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super(AdaptiveEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, self.num_models),
            nn.Softmax(dim=1),
        )
    
    def forward(
        self,
        inputs: Union[Dict, torch.Tensor],
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        individual_predictions = []
        
        for i, model in enumerate(self.models):
            model.eval() if not self.training else model.train()
            
            if isinstance(inputs, dict):
                pred = model(inputs['graph'])
                if hasattr(model, 'get_graph_embedding') and features is None:
                    if i == 0:  
                        features = model.get_graph_embedding(inputs['graph'])
            else:
                pred = model(inputs)
                if features is None and i == 0:
                    features = inputs.mean(dim=1) if len(inputs.shape) > 2 else inputs
        
        if features is None:
            weights = torch.ones(self.num_models, device=individual_predictions[0].device) / self.num_models
            weights = weights.unsqueeze(0).expand(individual_predictions[0].shape[0], -1)
        else:
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            if features.shape[-1] != self.weight_predictor[0].in_features:
                projection = nn.Linear(features.shape[-1], self.weight_predictor[0].in_features).to(features.device)
                features = projection(features)
            
            weights = self.weight_predictor(features)
        
        stacked = torch.stack(individual_predictions, dim=1)
        ensemble_pred = (stacked * weights).sum(dim=1)
        
        return ensemble_pred

