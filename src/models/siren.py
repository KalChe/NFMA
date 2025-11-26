# siren neural field implementation
import torch
import torch.nn as nn
import numpy as np

from ..config import SIREN_HIDDEN_DIM, SIREN_NUM_LAYERS, SIREN_OMEGA_0


class SIREN(nn.Module):
    # siren neural field with sinusoidal activations
    # reference: sitzmann et al., implicit neural representations (neursips 2020)
    def __init__(
        self, 
        input_dim: int = 2, 
        hidden_dim: int = SIREN_HIDDEN_DIM, 
        output_dim: int = 1, 
        num_layers: int = SIREN_NUM_LAYERS, 
        omega_0: float = SIREN_OMEGA_0
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.layers = nn.ModuleList()
        # build network
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # apply siren-specific weight initialization
        with torch.no_grad():
            # init first layer uniform
            self.layers[0].weight.uniform_(
                -1 / self.layers[0].in_features, 
                1 / self.layers[0].in_features
            )
            # init hidden/output scaled
            for layer in self.layers[1:]:
                bound = np.sqrt(6 / layer.in_features) / self.omega_0
                layer.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass with sinusoidal activations
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:
                x = torch.sin(self.omega_0 * layer(x))
            else:
                x = torch.sin(layer(x))
        return self.layers[-1](x)
    
    def set_parameters(self, params: torch.Tensor):
        # set network parameters from a flat tensor
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data = params[idx:idx+n].reshape(p.shape)
            idx += n
    
    def get_num_params(self) -> int:
        # return total number of parameters
        return sum(p.numel() for p in self.parameters())
