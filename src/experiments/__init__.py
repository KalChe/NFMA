# experiments package
from .mnist import run_mnist_experiment, create_mnist_figure
from .pde import run_pde_experiment, create_pde_figure
from .attention_convergence import (
    run_attention_convergence_experiments,
    create_temperature_scaling_figure,
    run_convergence_experiment
)
from .rotation_equivariance import (
    run_rotation_equivariance_experiments,
    run_so2_equivariance,
    run_so3_equivariance,
    create_so2_figure,
    create_so3_figure
)
from .ablation import (
    run_ablation_experiments,
    run_ablation_study,
    create_ablation_figure
)

__all__ = [
    'run_mnist_experiment',
    'create_mnist_figure',
    'run_pde_experiment',
    'create_pde_figure',
    'run_attention_convergence_experiments',
    'create_temperature_scaling_figure',
    'run_convergence_experiment',
    'run_rotation_equivariance_experiments',
    'run_so2_equivariance',
    'run_so3_equivariance',
    'create_so2_figure',
    'create_so3_figure',
    'run_ablation_experiments',
    'run_ablation_study',
    'create_ablation_figure',
]
