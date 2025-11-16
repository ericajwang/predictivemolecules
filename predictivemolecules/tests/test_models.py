"""
Unit tests for model architectures.
"""

import torch
import pytest
from models import GNNModel, TransformerModel, CNNModel, EnsembleModel
from torch_geometric.data import Data, Batch


def test_gnn_model():
    """Test GNN model forward pass."""
    model = GNNModel(input_dim=44, hidden_dim=64, num_layers=2)
    
    # Create dummy graph
    x = torch.randn(10, 44)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_attr = torch.randn(5, 10)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([graph])
    
    # Forward pass
    output = model(batch)
    assert output.shape == (1,)


def test_transformer_model():
    """Test Transformer model forward pass."""
    model = TransformerModel(input_dim=2048, d_model=128, num_layers=2)
    
    # Create dummy input
    x = torch.randn(4, 2048)  # batch_size=4
    
    # Forward pass
    output = model(x)
    assert output.shape == (4,)


def test_cnn_model():
    """Test CNN model forward pass."""
    model = CNNModel(input_dim=2048, hidden_dims=[256, 128])
    
    # Create dummy input
    x = torch.randn(4, 2048)  # batch_size=4
    
    # Forward pass
    output = model(x)
    assert output.shape == (4,)


def test_ensemble_model():
    """Test Ensemble model forward pass."""
    # Create base models
    gnn = GNNModel(input_dim=44, hidden_dim=64, num_layers=2)
    transformer = TransformerModel(input_dim=2048, d_model=128, num_layers=2)
    cnn = CNNModel(input_dim=2048, hidden_dims=[256, 128])
    
    ensemble = EnsembleModel(
        models=[gnn, transformer, cnn],
        strategy='weighted_average',
    )
    
    # Test with fingerprint input (for transformer and cnn)
    x = torch.randn(4, 2048)
    output = ensemble(x)
    assert output.shape == (4,)

