import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class CNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: list = [512, 256, 128],
        kernel_sizes: list = [7, 5, 3],
        dropout: float = 0.2,
        use_residual: bool = True,
        pool_type: str = 'adaptive',
    ):
        super(CNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.pool_type = pool_type
        self.use_residual = use_residual
        self.input_proj = nn.Conv1d(1, hidden_dims[0], kernel_size=1)
        self.conv_layers = nn.ModuleList()

        for i in range(len(hidden_dims)):
            in_channels = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            out_channels = hidden_dims[i]
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            
            if use_residual and i > 0:
                layer = ResidualBlock1D(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            else:
                layer = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            
            self.conv_layers.append(layer)
        
        if pool_type == 'adaptive':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
        
        final_dim = hidden_dims[-1]
        self.output_layers = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, final_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 4, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Predictions [batch_size, 1]
        """
        # Reshape for 1D convolution: [batch_size, 1, input_dim]
        x = x.unsqueeze(1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Pooling
        x = self.pool(x)  # [batch_size, channels, 1]
        x = x.squeeze(-1)  # [batch_size, channels]
        
        # Output prediction
        output = self.output_layers(x)
        
        return output.squeeze(-1) if output.shape[-1] == 1 else output


class MultiFingerprintCNN(nn.Module):
    """
    CNN model that processes multiple fingerprint types.
    """
    
    def __init__(
        self,
        fingerprint_dims: dict = {'ECFP': 2048, 'MACCS': 2048, 'RDKit': 2048},
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        Initialize multi-fingerprint CNN.
        
        Args:
            fingerprint_dims: Dictionary mapping fingerprint types to dimensions
            hidden_dim: Hidden dimension for fusion
            dropout: Dropout rate
        """
        super(MultiFingerprintCNN, self).__init__()
        
        # Individual CNN encoders for each fingerprint type
        self.encoders = nn.ModuleDict()
        for fp_type, fp_dim in fingerprint_dims.items():
            self.encoders[fp_type] = CNNModel(
                input_dim=fp_dim,
                hidden_dims=[512, 256, 128],
                dropout=dropout,
            )
        
        # Fusion layer
        num_fps = len(fingerprint_dims)
        self.fusion = nn.Sequential(
            nn.Linear(128 * num_fps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, fingerprints: dict) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            fingerprints: Dictionary of fingerprint tensors
            
        Returns:
            Predictions [batch_size, 1]
        """
        # Encode each fingerprint type
        encoded = []
        for fp_type, encoder in self.encoders.items():
            if fp_type in fingerprints:
                # Get embeddings from encoder (before final output layer)
                fp = fingerprints[fp_type]
                # Pass through CNN layers
                fp = fp.unsqueeze(1)
                fp = encoder.input_proj(fp)
                for layer in encoder.conv_layers:
                    fp = layer(fp)
                fp = encoder.pool(fp).squeeze(-1)
                encoded.append(fp)
        
        # Concatenate all encodings
        if encoded:
            fused = torch.cat(encoded, dim=1)
        else:
            # Fallback if no fingerprints provided
            fused = torch.zeros(fingerprints[list(fingerprints.keys())[0]].shape[0], 128 * len(self.encoders))
            if fused.is_cuda:
                fused = fused.cuda()
        
        # Final prediction
        output = self.fusion(fused)
        
        return output.squeeze(-1) if output.shape[-1] == 1 else output

