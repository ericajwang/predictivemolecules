"""
Graph Neural Network model for molecular binding prediction.
Uses molecular graph structure with node and edge features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    BatchNorm,
)
from torch_geometric.data import Batch
from typing import Optional, List


class GNNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        conv_type: str = 'GCN',
        num_heads: int = 1,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super(GNNLayer, self).__init__()
        
        self.conv_type = conv_type
        self.use_batch_norm = use_batch_norm
        
        if conv_type == 'GCN':
            self.conv = GCNConv(in_dim, out_dim)
        elif conv_type == 'GAT':
            self.conv = GATConv(
                in_dim, out_dim, heads=num_heads, dropout=dropout, concat=True
            )
            if num_heads > 1:
                self.projection = nn.Linear(num_heads * out_dim, out_dim)
            else:
                self.projection = nn.Identity()
        elif conv_type == 'GIN':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            self.conv = GINConv(mlp, train_eps=True)
        else:
            raise ValueError(f"Unknown convolution type: {conv_type}")
        
        if use_batch_norm:
            self.batch_norm = BatchNorm(out_dim)
        else:
            self.batch_norm = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.conv_type == 'GAT':
            x = self.conv(x, edge_index)
            x = self.projection(x)
        elif self.conv_type == 'GIN':
            x = self.conv(x, edge_index)
        else:
            x = self.conv(x, edge_index)
        
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class GNNModel(nn.Module):
    Grap    def __init__(
        self,
        input_dim: int = 44,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,
        conv_type: str = 'GAT',
        num_heads: int = 4,
        dropout: float = 0.2,
        pooling: str = 'mean',
        use_edge_features: bool = True,
        edge_dim: int = 10,
    ):
        """
        input_dim: dim of input node features
        hidden_dim: dim of hidden layers
        num_layers: num of GNN layers
        output_dim: dim of output (1 for regression)
        conv_type: type of convolution ('GCN', 'GAT', 'GIN')
        num_heads: num of attention heads for GAT
        dropout: Dropout rate
        pooling: Pooling strategy ('mean', 'max', 'sum', 'attention')
        use_edge_features: whether to use edge features
        edge_dim: dim of edge features
        """
        super(GNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.use_edge_features = use_edge_features
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GNNLayer(
                    hidden_dim,
                    hidden_dim,
                    conv_type=conv_type,
                    num_heads=num_heads if i == 0 else 1, 
                    dropout=dropout,
                    use_batch_norm=True,
                )
            )
        
        if use_edge_features:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim),
        )
    
    def forward(self, batch: Batch) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        x = self.input_proj(x)
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
        
        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(x, batch_idx)
        elif self.pooling == 'max':
            graph_embedding = global_max_pool(x, batch_idx)
        elif self.pooling == 'sum':
            graph_embedding = global_add_pool(x, batch_idx)
        elif self.pooling == 'attention':
            attention_weights = self.attention_pool(x)
            attention_weights = F.softmax(attention_weights, dim=0)
            graph_embedding = (x * attention_weights).sum(dim=0)
            graph_embedding = graph_embedding.unsqueeze(0).repeat(batch_idx.max().item() + 1, 1)
        else:
            graph_embedding = global_mean_pool(x, batch_idx)
        
        output = self.output_layers(graph_embedding)
        
        return output.squeeze(-1) if output.shape[-1] == 1 else output
    
    def get_graph_embedding(self, batch: Batch) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        x = self.input_proj(x)
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
        
        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(x, batch_idx)
        elif self.pooling == 'max':
            graph_embedding = global_max_pool(x, batch_idx)
        elif self.pooling == 'sum':
            graph_embedding = global_add_pool(x, batch_idx)
        else:
            graph_embedding = global_mean_pool(x, batch_idx)
        
        return graph_embedding

