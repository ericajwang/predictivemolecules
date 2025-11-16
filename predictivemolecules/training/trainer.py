"""
Training utilities and trainer class for model training.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Callable
import json
from datetime import datetime

from ..evaluation.metrics import calculate_metrics, MetricTracker


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        log_dir: str = './logs',
        save_best: bool = True,
        patience: int = 10,
        min_delta: float = 0.001,
    ):
        """
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            save_best: Whether to save best model
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.save_best = save_best
        self.patience = patience
        self.min_delta = min_delta
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=0.001,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, timestamp))
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.metric_tracker = MetricTracker()
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, dict):
                if 'graph' in batch:
                    batch['graph'] = batch['graph'].to(self.device)
                if 'fingerprint' in batch:
                    batch['fingerprint'] = batch['fingerprint'].to(self.device)
                targets = batch['target'].to(self.device)
            else:
                batch = batch.to(self.device)
                targets = batch.target.to(self.device)

            self.optimizer.zero_grad()
            
            if isinstance(batch, dict):
                if 'graph' in batch:
                    predictions = self.model(batch['graph'])
                elif 'fingerprint' in batch:
                    predictions = self.model(batch['fingerprint'])
                else:
                    predictions = self.model(batch)
            else:
                predictions = self.model(batch)
            
            loss = self.criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                if isinstance(batch, dict):
                    if 'graph' in batch:
                        batch['graph'] = batch['graph'].to(self.device)
                    if 'fingerprint' in batch:
                        batch['fingerprint'] = batch['fingerprint'].to(self.device)
                    targets = batch['target'].to(self.device)
                else:
                    batch = batch.to(self.device)
                    targets = batch.target.to(self.device)
                
                # Forward pass
                if isinstance(batch, dict):
                    if 'graph' in batch:
                        predictions = self.model(batch['graph'])
                    elif 'fingerprint' in batch:
                        predictions = self.model(batch['fingerprint'])
                    else:
                        predictions = self.model(batch)
                else:
                    predictions = self.model(batch)
                
                loss = self.criterion(predictions, targets)
                
                # Accumulate metrics
                total_loss += loss.item()
                all_predictions.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                if isinstance(batch, dict):
                    if 'graph' in batch:
                        batch['graph'] = batch['graph'].to(self.device)
                    if 'fingerprint' in batch:
                        batch['fingerprint'] = batch['fingerprint'].to(self.device)
                    targets = batch['target'].to(self.device)
                else:
                    batch = batch.to(self.device)
                    targets = batch.target.to(self.device)
                
                if isinstance(batch, dict):
                    if 'graph' in batch:
                        predictions = self.model(batch['graph'])
                    elif 'fingerprint' in batch:
                        predictions = self.model(batch['fingerprint'])
                    else:
                        predictions = self.model(batch)
                else:
                    predictions = self.model(batch)
                
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(self, epochs: int = 100, verbose: bool = True):
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['loss'])
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()

            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"RMSE: {train_metrics.get('rmse', 0):.4f}, "
                      f"MAE: {train_metrics.get('mae', 0):.4f}, "
                      f"R²: {train_metrics.get('r2', 0):.4f}")
                
                if val_metrics:
                    print(f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"RMSE: {val_metrics.get('rmse', 0):.4f}, "
                          f"MAE: {val_metrics.get('mae', 0):.4f}, "
                          f"R²: {val_metrics.get('r2', 0):.4f}")
            
            if self.save_best and val_metrics:
                val_loss = val_metrics['loss']
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt', epoch, val_metrics)
                else:
                    self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\nearly stopping at epoch {epoch + 1}")
                break
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"checkpoint loaded from {filepath}")

