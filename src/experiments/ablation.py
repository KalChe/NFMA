# ablation study: affine equivariance under different model configurations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..config import (
    DEVICE, SEED, LEARNING_RATE, WEIGHT_DECAY, NUM_ITERATIONS,
    GRID_SIZE, GRID_RANGE, TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_DROPOUT, AFFINE_SCALE, AFFINE_TRANSLATION
)
from ..models import SIREN
from ..utils import set_seed, save_figure, load_mnist_field, create_grid


class AblationOperator(nn.Module):
    # configurable transformer operator for ablation study
    def __init__(
        self,
        embed_dim: int = TRANSFORMER_EMBED_DIM,
        num_heads: int = TRANSFORMER_NUM_HEADS,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dropout: float = TRANSFORMER_DROPOUT,
        target_model_params: int = None,
        use_normalization: bool = True,
        use_absolute_pe: bool = False,
        temperature: float = 1.0
    ):
        super().__init__()
        self.use_normalization = use_normalization
        self.use_absolute_pe = use_absolute_pe
        self.temperature = temperature
        
        input_dim = 3
        if use_absolute_pe:
            # add sinusoidal positional encoding channels
            self.pe_dim = 8
            input_dim = 3 + self.pe_dim * 2  # sin + cos for each freq
        
        self.input_embed = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if target_model_params is None:
            target_model_params = 2*64 + 64 + 64*64 + 64 + 64*64 + 64 + 64*1 + 1
        
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, target_model_params)
        )
    
    def _positional_encoding(self, coords):
        # sinusoidal absolute positional encoding
        freqs = 2 ** torch.arange(self.pe_dim, device=coords.device, dtype=coords.dtype)
        # coords: (batch, n_points, 2) -> pe: (batch, n_points, pe_dim*2)
        angles = coords.unsqueeze(-1) * freqs  # (batch, n_points, 2, pe_dim)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, n_points, 2, pe_dim*2)
        return pe.reshape(coords.shape[0], coords.shape[1], -1)[:, :, :self.pe_dim * 2]
    
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        coords = S[:, :, :2]
        values = S[:, :, 2:3]
        
        if self.use_normalization:
            # affine-preserving normalization
            mu = coords.mean(dim=1, keepdim=True)
            centered = coords - mu
            scale = torch.sqrt((centered**2).mean(dim=[1, 2], keepdim=True)).clamp(min=1e-6)
            coords_proc = centered / scale
        else:
            # no normalization - use raw coordinates
            coords_proc = coords
        
        if self.use_absolute_pe:
            pe = self._positional_encoding(coords_proc)
            tokens = torch.cat([coords_proc, values, pe], dim=-1)
        else:
            tokens = torch.cat([coords_proc, values], dim=-1)
        
        embedded = self.input_embed(tokens)
        
        # apply temperature scaling to attention (via scaling embeddings)
        if self.temperature != 1.0:
            embedded = embedded / np.sqrt(self.temperature)
        
        out = self.transformer(embedded)
        pooled = out.mean(dim=1)
        return self.output_head(pooled)


def train_ablation_operator(operator, data_loader_fn, num_iterations=NUM_ITERATIONS):
    # train operator for ablation study
    optimizer = optim.AdamW(operator.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    
    for it in range(num_iterations):
        result = data_loader_fn()
        if isinstance(result, tuple):
            samples, _ = result
        else:
            samples = result
        
        S = samples.unsqueeze(0)
        pred_params = operator(S)
        pred_siren = SIREN().to(DEVICE)
        pred_siren.set_parameters(pred_params[0])
        
        with torch.no_grad():
            y_true = S[0, :, 2:3]
        y_pred = pred_siren(S[0, :, :2])
        loss = nn.functional.mse_loss(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def evaluate_equivariance(operator, data_loader_fn, scale=AFFINE_SCALE, translation=AFFINE_TRANSLATION):
    # evaluate affine equivariance: |f_S'(x) - f_S((x-b)/a)|
    result = data_loader_fn()
    if isinstance(result, tuple):
        samples, _ = result
    else:
        samples = result
    
    coords = samples[:, :2]
    values = samples[:, 2:3]
    b = torch.tensor(translation, device=DEVICE).float()
    
    # transform coordinates: S' = aS + b
    coords_t = coords * scale + b
    samples_t = torch.cat([coords_t, values], dim=-1)
    
    with torch.no_grad():
        # get f_S from original samples
        params1 = operator(samples.unsqueeze(0))
        f_S = SIREN().to(DEVICE)
        f_S.set_parameters(params1[0])
        
        # get f_S' from transformed samples
        params2 = operator(samples_t.unsqueeze(0))
        f_Sp = SIREN().to(DEVICE)
        f_Sp.set_parameters(params2[0])
    
    grid = create_grid(GRID_SIZE, GRID_RANGE[0], GRID_RANGE[1])
    
    with torch.no_grad():
        # f_S'(x)
        field_Sp = f_Sp(grid).reshape(GRID_SIZE, GRID_SIZE).cpu().numpy()
        # f_S((x-b)/a)
        field_S_inv = f_S((grid - b) / scale).reshape(GRID_SIZE, GRID_SIZE).cpu().numpy()
        # equivariance error: |f_S'(x) - f_S((x-b)/a)|
        error_field = np.abs(field_Sp - field_S_inv)
    
    mse = np.mean(error_field**2)
    mae = np.mean(error_field)
    max_err = np.max(error_field)
    
    return {
        'field_Sp': field_Sp,
        'field_S_inv': field_S_inv,
        'error_field': error_field,
        'mse': mse,
        'mae': mae,
        'max_err': max_err,
        'samples_orig': samples.cpu().numpy(),
        'samples_trans': samples_t.cpu().numpy()
    }


def run_ablation_study(num_iterations=150, verbose=True):
    # run ablation study with different configurations (single seed, for figure)
    set_seed(SEED)
    
    configs = {
        'Full (Norm + High τ)': {'use_normalization': True, 'use_absolute_pe': False, 'temperature': 10.0},
        'No Normalization': {'use_normalization': False, 'use_absolute_pe': False, 'temperature': 10.0},
        'Absolute PE': {'use_normalization': True, 'use_absolute_pe': True, 'temperature': 10.0},
        'Low Temperature': {'use_normalization': True, 'use_absolute_pe': False, 'temperature': 0.1},
    }
    
    results = {}
    siren_params = SIREN().get_num_params()
    
    for name, cfg in configs.items():
        if verbose:
            print(f"\n  training {name}...")
        
        set_seed(SEED)
        operator = AblationOperator(
            target_model_params=siren_params,
            use_normalization=cfg['use_normalization'],
            use_absolute_pe=cfg['use_absolute_pe'],
            temperature=cfg['temperature']
        ).to(DEVICE)
        
        losses = train_ablation_operator(operator, load_mnist_field, num_iterations=num_iterations)
        
        set_seed(SEED)  # reset for consistent evaluation
        eval_result = evaluate_equivariance(operator, load_mnist_field)
        eval_result['losses'] = losses
        eval_result['config'] = cfg
        results[name] = eval_result
        
        if verbose:
            print(f"    MSE: {eval_result['mse']:.6f}, MAE: {eval_result['mae']:.6f}, Max: {eval_result['max_err']:.6f}")
    
    return results


def run_ablation_multi_seed(num_seeds=10, num_iterations=150, verbose=True):
    # run ablation study with multiple seeds for statistical table
    configs = {
        'Linear Attention': {'use_normalization': True, 'use_absolute_pe': False, 'temperature': 1.0, 'linear': True},
        'Softmax (τ = 1)': {'use_normalization': True, 'use_absolute_pe': False, 'temperature': 1.0, 'linear': False},
        'Softmax (τ = 100)': {'use_normalization': True, 'use_absolute_pe': False, 'temperature': 100.0, 'linear': False},
        'No Normalization': {'use_normalization': False, 'use_absolute_pe': False, 'temperature': 1.0, 'linear': False},
        'Absolute Pos. Enc.': {'use_normalization': True, 'use_absolute_pe': True, 'temperature': 1.0, 'linear': False},
    }
    
    siren_params = SIREN().get_num_params()
    all_results = {name: [] for name in configs}
    
    for seed_idx in range(num_seeds):
        current_seed = SEED + seed_idx
        if verbose:
            print(f"\n  Seed {seed_idx + 1}/{num_seeds} (seed={current_seed})")
        
        for name, cfg in configs.items():
            set_seed(current_seed)
            
            operator = AblationOperator(
                target_model_params=siren_params,
                use_normalization=cfg['use_normalization'],
                use_absolute_pe=cfg['use_absolute_pe'],
                temperature=cfg['temperature']
            ).to(DEVICE)
            
            train_ablation_operator(operator, load_mnist_field, num_iterations=num_iterations)
            
            set_seed(current_seed)
            eval_result = evaluate_equivariance(operator, load_mnist_field)
            all_results[name].append(eval_result['mse'])
            
            if verbose:
                print(f"    {name}: MSE = {eval_result['mse']:.2e}")
    
    # compute statistics
    stats = {}
    for name in configs:
        mse_vals = np.array(all_results[name])
        stats[name] = {
            'mean': np.mean(mse_vals),
            'std': np.std(mse_vals),
            'values': mse_vals
        }
    
    return stats


def format_sci_notation(mean, std):
    # format as (mean ± std) × 10^exp
    if mean == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(mean))))
    mantissa_mean = mean / (10 ** exp)
    mantissa_std = std / (10 ** exp)
    return f"({mantissa_mean:.1f} ± {mantissa_std:.1f}) × 10^{exp}"


def print_ablation_table(stats, verbose=True):
    # print LaTeX-style table of ablation results
    if verbose:
        print("\n" + "="*60)
        print("Ablation Study: Affine Equivariance Error (MSE)")
        print("="*60)
        print(f"{'Configuration':<25} {'Affine Error':<25}")
        print("-"*60)
        
        for name, s in stats.items():
            formatted = format_sci_notation(s['mean'], s['std'])
            print(f"{name:<25} {formatted:<25}")
        
        print("="*60)
        
        # also print LaTeX table format
        print("\nLaTeX Table Format:")
        print("\\begin{tabular}{lc}")
        print("\\toprule")
        print("Configuration & Affine Error \\\\")
        print("\\midrule")
        for name, s in stats.items():
            exp = int(np.floor(np.log10(abs(s['mean'])))) if s['mean'] != 0 else 0
            mantissa_mean = s['mean'] / (10 ** exp)
            mantissa_std = s['std'] / (10 ** exp)
            print(f"{name} & $({mantissa_mean:.1f} \\pm {mantissa_std:.1f}) \\times 10^{{{exp}}}$ \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    
    return stats


def create_error_grid_figure(results, verbose=True):
    # create 2x2 grid of error (MSE) fields only
    if verbose:
        print("\n  creating error grid figure...")
    
    config_names = list(results.keys())
    
    # find global error range
    all_errors = [results[name]['error_field'] for name in config_names]
    vmax_error = max(e.max() for e in all_errors)
    
    extent = [GRID_RANGE[0], GRID_RANGE[1], GRID_RANGE[0], GRID_RANGE[1]]
    
    # create figure with extra space on right for colorbar
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.25, hspace=0.3)
    
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    
    for idx, name in enumerate(config_names):
        res = results[name]
        ax = axes[idx]
        
        im = ax.imshow(res['error_field'], extent=extent, origin='lower',
                       cmap='hot', vmin=0, vmax=vmax_error)
        ax.set_title(f"{name}\nMSE = {res['mse']:.6f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.tick_params(labelsize=8)
    
    # colorbar on far right
    cbar_ax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$|f_{S\prime}(x) - f_S(\frac{x-b}{a})|$', fontsize=11)
    
    fig.suptitle('Equivariance Error: Ablation Study', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'ablation_error_grid')
    plt.close(fig)
    
    if verbose:
        print("  error grid figure saved")


def create_sample_distribution_figure(results, verbose=True):
    # create sample distribution figure: 1 row, 2 columns (original vs transformed) with heatmap
    if verbose:
        print("\n  creating sample distribution figure...")
    
    # use first config's samples (they're all the same since we reset seed)
    first_name = list(results.keys())[0]
    samples_orig = results[first_name]['samples_orig']
    samples_trans = results[first_name]['samples_trans']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # original samples S with intensity heatmap
    ax0 = axes[0]
    scatter0 = ax0.scatter(samples_orig[:, 0], samples_orig[:, 1], 
                           c=samples_orig[:, 2], cmap='viridis', s=8, alpha=0.8)
    ax0.set_xlabel('x', fontsize=12)
    ax0.set_ylabel('y', fontsize=12)
    ax0.set_title('Original Samples S', fontsize=13, fontweight='bold')
    ax0.set_aspect('equal')
    ax0.grid(True, alpha=0.3)
    cbar0 = fig.colorbar(scatter0, ax=ax0, shrink=0.8)
    cbar0.set_label('Field Value', fontsize=10)
    
    # transformed samples S' with intensity heatmap
    ax1 = axes[1]
    scatter1 = ax1.scatter(samples_trans[:, 0], samples_trans[:, 1],
                           c=samples_trans[:, 2], cmap='viridis', s=8, alpha=0.8)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title("Transformed Samples S' = aS + b", fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label('Field Value', fontsize=10)
    
    fig.suptitle('Sample Point Distribution: Pre and Post Affine Transformation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, 'ablation_sample_distribution')
    plt.close(fig)
    
    if verbose:
        print("  sample distribution figure saved")


def create_ablation_figure(results, verbose=True):
    # create ablation figure (error grid only)
    if verbose:
        print("\n  creating ablation figures...")
    
    # only the 2x2 error grid
    create_error_grid_figure(results, verbose=False)
    
    if verbose:
        print("  ablation figures saved")
    
    return None


# legacy function kept for compatibility
def run_ablation_experiments(verbose=True):
    # main entry point for ablation study
    if verbose:
        print("\n  running ablation study...")
    
    # run multi-seed study for table
    stats = run_ablation_multi_seed(num_seeds=10, num_iterations=150, verbose=verbose)
    print_ablation_table(stats, verbose=verbose)
    
    # run single-seed study for figure (using first config set for visualization)
    results = run_ablation_study(num_iterations=150, verbose=False)
    create_ablation_figure(results, verbose=verbose)
    
    return stats
