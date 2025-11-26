# training utilities
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_ITERATIONS,
    GRID_SIZE, GRID_RANGE
)
from ..models import SIREN
from .data import create_grid


def train_operator(
    operator: nn.Module, 
    data_loader_fn,
    num_iterations: int = NUM_ITERATIONS, 
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    verbose: bool = False
) -> list:
    # train a transformer operator on neural field fitting
    optimizer = optim.AdamW(operator.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    
    for it in range(num_iterations):
        result = data_loader_fn()
        if isinstance(result, tuple):
            samples, _ = result
        else:
            samples = result
        
        S = samples.unsqueeze(0)
        
        
        # Forward pass
        pred_params = operator(S)
        pred_siren = SIREN().to(DEVICE)
        pred_siren.set_parameters(pred_params[0])
        
        # Compute loss
        with torch.no_grad():
            y_true = S[0, :, 2:3]
        y_pred = pred_siren(S[0, :, :2])
        loss = nn.functional.mse_loss(y_pred, y_true)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (it + 1) % 50 == 0:
            print(f"  Iteration {it+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    return losses


def validate_affine_equivariance(
    operator: nn.Module,
    data_loader_fn,
    scale: float = 1.5,
    translation: tuple = (0.3, -0.3),
    grid_size: int = GRID_SIZE,
    grid_range: tuple = GRID_RANGE
) -> tuple:
    # validate affine equivariance of the operator; returns (mse, field_A, field_B)
    # Load data
    result = data_loader_fn()
    if isinstance(result, tuple):
        samples, _ = result
    else:
        samples = result
    
    coords = samples[:, :2]
    values = samples[:, 2:3]
    b = torch.tensor(translation, device=DEVICE).float()
    
    # Apply affine transformation to coordinates
    coords_t = coords * scale + b
    samples_t = torch.cat([coords_t, values], dim=-1)
    
    # Get predictions for both
    with torch.no_grad():
        params1 = operator(samples.unsqueeze(0))
        f_S = SIREN().to(DEVICE)
        f_S.set_parameters(params1[0])
        
        params2 = operator(samples_t.unsqueeze(0))
        f_Sp = SIREN().to(DEVICE)
        f_Sp.set_parameters(params2[0])
    
    # Create evaluation grid
    grid = create_grid(grid_size, grid_range[0], grid_range[1])
    
    # Evaluate fields
    with torch.no_grad():
        # f_S'(x) - field fitted to transformed samples
        field_A = f_Sp(grid).reshape(grid_size, grid_size).cpu().numpy()
        # f_S((x-b)/a) - original field with inverse transform
        field_B = f_S((grid - b) / scale).reshape(grid_size, grid_size).cpu().numpy()
    
    mse = np.mean((field_A - field_B)**2)
    return mse, field_A, field_B
