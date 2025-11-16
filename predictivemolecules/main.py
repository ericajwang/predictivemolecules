import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.config_loader import load_config
from data.data_loader import MoleculeDataset, MoleculeDataLoader
from models import GNNModel, TransformerModel, CNNModel, EnsembleModel
from training.trainer import Trainer
from training.losses import MSELoss, MAELoss, HuberLoss, CombinedLoss
from training.optimizers import get_optimizer, get_scheduler
from utils.model_utils import count_parameters, get_model_summary
from utils.data_utils import train_val_test_split
import pandas as pd


def create_model(config):
    # creates model based on configuration
    model_config = config.model
    
    if model_config['type'] == 'gnn':
        model = GNNModel(
            input_dim=model_config.get('input_dim', 44),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 3),
            output_dim=model_config.get('output_dim', 1),
            conv_type=model_config.get('conv_type', 'GAT'),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.2),
            pooling=model_config.get('pooling', 'mean'),
            use_edge_features=model_config.get('use_edge_features', True),
            edge_dim=model_config.get('edge_dim', 10),
        )
    
    elif model_config['type'] == 'transformer':
        model = TransformerModel(
            input_dim=model_config.get('input_dim', 2048),
            d_model=model_config.get('d_model', 256),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 4),
            d_ff=model_config.get('d_ff', 1024),
            max_seq_len=model_config.get('max_seq_len', 5000),
            dropout=model_config.get('dropout', 0.1),
            pooling=model_config.get('pooling', 'mean'),
        )
    
    elif model_config['type'] == 'cnn':
        model = CNNModel(
            input_dim=model_config.get('input_dim', 2048),
            hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
            kernel_sizes=model_config.get('kernel_sizes', [7, 5, 3]),
            dropout=model_config.get('dropout', 0.2),
            use_residual=model_config.get('use_residual', True),
            pool_type=model_config.get('pool_type', 'adaptive'),
        )
    
    elif model_config['type'] == 'ensemble':
        # Create base models
        base_models = []
        for base_model_config in model_config.get('base_models', []):
            if base_model_config['type'] == 'gnn':
                base_model = GNNModel(
                    hidden_dim=base_model_config.get('hidden_dim', 128),
                    num_layers=base_model_config.get('num_layers', 3),
                    conv_type=base_model_config.get('conv_type', 'GAT'),
                )
            elif base_model_config['type'] == 'transformer':
                base_model = TransformerModel(
                    d_model=base_model_config.get('d_model', 256),
                    num_layers=base_model_config.get('num_layers', 4),
                )
            elif base_model_config['type'] == 'cnn':
                base_model = CNNModel(
                    hidden_dims=base_model_config.get('hidden_dims', [512, 256, 128]),
                )
            else:
                raise ValueError(f"Unknown base model type: {base_model_config['type']}")
            
            base_models.append(base_model)
        
        model = EnsembleModel(
            models=base_models,
            strategy=model_config.get('strategy', 'weighted_average'),
            learnable_weights=model_config.get('learnable_weights', True),
        )
    
    else:
        raise ValueError(f"unknown model type {model_config['type']}")
    
    return model


def create_loss(config):
    # creates loss function based on configuration
    loss_config = config.loss
    loss_type = loss_config.get('type', 'mse')
    
    if loss_type == 'mse':
        return MSELoss()
    elif loss_type == 'mae':
        return MAELoss()
    elif loss_type == 'huber':
        delta = loss_config.get('delta', 1.0)
        return HuberLoss(delta=delta)
    elif loss_type == 'combined':
        loss_types = loss_config.get('loss_types', ['mse', 'mae'])
        weights = loss_config.get('weights', None)
        return CombinedLoss(loss_types=loss_types, weights=weights)
    else:
        return MSELoss()


def main():
    parser = argparse.ArgumentParser(description='train molecular binding prediction model')
    parser.add_argument('--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = config.training.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    data_config = config.data
    if data_config.get('data_path') and pd.io.common.file_exists(data_config['data_path']):
        df = pd.read_csv(data_config['data_path'])
        train_df, val_df, test_df = train_val_test_split(
            df,
            train_ratio=data_config.get('train_split', 0.7),
            val_ratio=data_config.get('val_split', 0.15),
            test_ratio=data_config.get('test_split', 0.15),
        )
        train_dataset = MoleculeDataset(
            data_path=None,
            smiles_column=data_config['smiles_column'],
            target_column=data_config['target_column'],
            mode=data_config.get('mode', 'graph'),
        )
        train_dataset.data = train_df
        train_dataset._preprocess_all()
        
        val_dataset = MoleculeDataset(
            data_path=None,
            smiles_column=data_config['smiles_column'],
            target_column=data_config['target_column'],
            mode=data_config.get('mode', 'graph'),
        )
        val_dataset.data = val_df
        val_dataset._preprocess_all()
        
        test_dataset = MoleculeDataset(
            data_path=None,
            smiles_column=data_config['smiles_column'],
            target_column=data_config['target_column'],
            mode=data_config.get('mode', 'graph'),
        )
        test_dataset.data = test_df
        test_dataset._preprocess_all()

        train_loader = MoleculeDataLoader.create_loader(
            train_dataset,
            batch_size=data_config.get('batch_size', 32),
            shuffle=data_config.get('shuffle', True),
            num_workers=data_config.get('num_workers', 0),
        )
        
        val_loader = MoleculeDataLoader.create_loader(
            val_dataset,
            batch_size=data_config.get('batch_size', 32),
            shuffle=False,
            num_workers=data_config.get('num_workers', 0),
        )
        
        test_loader = MoleculeDataLoader.create_loader(
            test_dataset,
            batch_size=data_config.get('batch_size', 32),
            shuffle=False,
            num_workers=data_config.get('num_workers', 0),
        )
    else:
        print("data file not found")
        train_loader = None
        val_loader = None
        test_loader = None
    
    model = create_model(config)
    print(get_model_summary(model))
    print(f"total parameters {count_parameters(model):,}")
    
    criterion = create_loss(config)
    
    optimizer = get_optimizer(
        model,
        optimizer_type=config.optimizer.get('type', 'adam'),
        lr=config.optimizer.get('lr', 0.001),
        weight_decay=config.optimizer.get('weight_decay', 1e-5),
    )
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=config.scheduler.get('type', 'reduce_on_plateau'),
        **{k: v for k, v in config.scheduler.items() if k != 'type'},
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config.training.get('save_dir', './checkpoints'),
        log_dir=config.training.get('log_dir', './logs'),
        save_best=config.training.get('save_best', True),
        patience=config.training.get('patience', 10),
        min_delta=config.training.get('min_delta', 0.001),
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(epochs=config.training.get('epochs', 100))
    
    if test_loader is not None:
        test_metrics = trainer.test()
        print("\ntest metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()

