# pde experiment
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ..config import DEVICE, SEED, NUM_TRIALS, RESULTS_DIR
from ..models import SIREN, TransformerOperator
from ..utils import (
    set_seed, solve_poisson_2d, train_operator,
    validate_affine_equivariance, save_figure
)


def load_pde_field():
    # wrapper for solve_poisson_2d matching data loader interface
    return solve_poisson_2d()


def run_pde_experiment(
    n_trials: int = NUM_TRIALS,
    num_iterations: int = 200,
    verbose: bool = True
) -> dict:
    # run pde neural field experiment; returns dictionary with mse stats and fields
    siren_params = SIREN().get_num_params()
    
    mses = []
    all_field_A = []
    all_field_B = []
    
    for t in range(n_trials):
        if verbose:
            print(f"\n  Trial {t+1}/{n_trials}")

        set_seed(SEED + t)

        operator = TransformerOperator(target_model_params=siren_params).to(DEVICE)
        train_operator(operator, load_pde_field, num_iterations=num_iterations, verbose=False)

        mse, field_A, field_B = validate_affine_equivariance(operator, load_pde_field)
        
        mses.append(mse)
        all_field_A.append(field_A)
        all_field_B.append(field_B)
        
        if verbose:
            print(f"    MSE: {mse:.6f}")
    
    # compute statistics
    mean_mse = np.mean(mses)
    sem = stats.sem(mses) if len(mses) > 1 else 0
    ci95 = sem * stats.t.ppf(0.975, max(n_trials - 1, 1))
    
    # Average fields for visualization
    avg_field_A = np.mean(np.stack(all_field_A), axis=0)
    avg_field_B = np.mean(np.stack(all_field_B), axis=0)
    
    return {
        'mses': mses,
        'mean_mse': mean_mse,
        'ci95': ci95,
        'avg_field_A': avg_field_A,
        'avg_field_B': avg_field_B
    }


def create_pde_figure(results: dict, save: bool = True, verbose: bool = True):
    # create publication figure for pde experiment
    diff = np.abs(results['avg_field_A'] - results['avg_field_B'])
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im0 = axes[0].imshow(results['avg_field_A'], cmap='viridis')
    axes[0].set_title("Transformed Input f_S'(x)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(results['avg_field_B'], cmap='viridis')
    axes[1].set_title("Original Inverse-Transformed f_S((x-b)/a)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f"Difference (MSE={results['mean_mse']:.4f}±{results['ci95']:.4f})")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    fig.suptitle('2D PDE Affine Equivariance Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'pde_affine_equivariance')
    
    plt.close(fig)
    return fig
