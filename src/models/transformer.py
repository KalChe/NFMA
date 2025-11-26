# transformer operator
import torch
import torch.nn as nn

from ..config import (
    TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS, 
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT
)


class TransformerOperator(nn.Module):
    # transformer operator for meta-learning neural fields
    # maps observations {(x_i, y_i)} -> siren parameters; preserves affine structure
    def __init__(
        self, 
        embed_dim: int = TRANSFORMER_EMBED_DIM, 
        num_heads: int = TRANSFORMER_NUM_HEADS, 
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dropout: float = TRANSFORMER_DROPOUT,
        target_model_params: int = None
    ):
        super().__init__()
        
        # input embedding (x,y,value) -> embed
        self.input_embed = nn.Linear(3, embed_dim)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            activation='gelu', 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # output head param count default
        if target_model_params is None:
            # Default: SIREN with hidden_dim=64, num_layers=3
            target_model_params = 2*64 + 64 + 64*64 + 64 + 64*64 + 64 + 64*1 + 1
        
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, target_model_params)
        )
    
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        # forward pass with affine-preserving coordinate normalization
        coords = S[:, :, :2]
        values = S[:, :, 2:3]
        
        # Affine-preserving normalization (Theorem 1)
        mu = coords.mean(dim=1, keepdim=True)
        centered = coords - mu
        scale = torch.sqrt((centered**2).mean(dim=[1, 2], keepdim=True)).clamp(min=1e-6)
        coords_norm = centered / scale
        
        # Combine normalized coords with values
        tokens = torch.cat([coords_norm, values], dim=-1)
        
        # Embed and transform
        embedded = self.input_embed(tokens)
        out = self.transformer(embedded)
        
        # Global pooling and output
        pooled = out.mean(dim=1)
        return self.output_head(pooled)
