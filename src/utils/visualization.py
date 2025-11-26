# visualization utilities
import numpy as np
import matplotlib.pyplot as plt

from ..config import RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT


def save_figure(fig, filename: str, dpi: int = FIGURE_DPI, 
                bbox_inches: str = 'tight', pad_inches: float = 0.1):
    # save figure to results dir
    filepath = RESULTS_DIR / f"{filename}.{FIGURE_FORMAT}"
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                pad_inches=pad_inches, facecolor='white')
    print(f"  Saved: {filepath}")


def plot_affine_comparison(
    field_A: np.ndarray, 
    field_B: np.ndarray, 
    mse: float,
    title: str = "Affine Equivariance Validation",
    save_name: str = None
):
    # plot comparison of transformed fields
    diff = np.abs(field_A - field_B)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(field_A)
    axes[0].set_title("f_S'(x)")
    axes[0].axis('off')
    
    axes[1].imshow(field_B)
    axes[1].set_title("f_S((x-b)/a)")
    axes[1].axis('off')
    
    axes[2].imshow(diff)
    axes[2].set_title(f"Diff (MSE={mse:.4f})")
    axes[2].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig
