import sys
import os
import argparse
import time

# add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SEED, DEVICE, RESULTS_DIR, 
    SIREN_OMEGA_0, SIREN_HIDDEN_DIM, SIREN_NUM_LAYERS,
    NUM_ITERATIONS, LEARNING_RATE
)
from src.utils import set_seed


def print_banner():
    print("=" * 70)
    print("Neural Fields Meet Attention (NFMA)")
    print("=" * 70)
    print(f"  Device:     {DEVICE}")
    print(f"  Seed:       {SEED}")
    print(f"  Results:    {RESULTS_DIR}")
    print(f"  SIREN:      ω₀={SIREN_OMEGA_0}, hidden={SIREN_HIDDEN_DIM}, layers={SIREN_NUM_LAYERS}")
    print(f"  Training:   iterations={NUM_ITERATIONS}, lr={LEARNING_RATE}")
    print("=" * 70)


def run_mnist(verbose: bool = True):
    # run mnist field reconstruction experiment
    if verbose:
        print("\n[1/4] MNIST Field Reconstruction")
        print("-" * 40)
    
    from src.experiments import run_mnist_experiment, create_mnist_figure
    
    start = time.time()
    results = run_mnist_experiment(verbose=verbose)
    create_mnist_figure(results, verbose=verbose)
    elapsed = time.time() - start
    
    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
    
    return results


def run_pde(verbose: bool = True):
    # run poisson pde solving experiment
    if verbose:
        print("\n[2/4] Poisson PDE Solving")
        print("-" * 40)
    
    from src.experiments import run_pde_experiment, create_pde_figure
    
    start = time.time()
    results = run_pde_experiment(verbose=verbose)
    create_pde_figure(results, verbose=verbose)
    elapsed = time.time() - start
    
    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
    
    return results


def run_attention(verbose: bool = True):
    # run attention convergence experiments
    if verbose:
        print("\n[3/4] Attention Convergence")
        print("-" * 40)
    
    from src.experiments import (
        run_attention_convergence_experiments,
        create_temperature_scaling_figure
    )
    
    start = time.time()
    summary = run_attention_convergence_experiments(verbose=verbose)
    create_temperature_scaling_figure(verbose=verbose)
    elapsed = time.time() - start
    
    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
    
    return summary


def run_rotation(verbose: bool = True):
    # run rotation equivariance experiments
    if verbose:
        print("\n[4/4] Rotation Equivariance")
        print("-" * 40)
    
    from src.experiments import run_rotation_equivariance_experiments
    
    start = time.time()
    so2_results, so3_results = run_rotation_equivariance_experiments(verbose=verbose)
    elapsed = time.time() - start
    
    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
    
    return so2_results, so3_results


def run_all(verbose: bool = True):
    # run all experiments
    print_banner()
    
    total_start = time.time()
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run experiments
    run_mnist(verbose)
    run_pde(verbose)
    run_attention(verbose)
    run_rotation(verbose)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print(f"All experiments completed in {total_elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 70)


def parse_args():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Neural Fields Meet Attention (NFMA) - Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--mnist', action='store_true',
                       help='Run MNIST field reconstruction experiment')
    parser.add_argument('--pde', action='store_true',
                       help='Run Poisson PDE solving experiment')
    parser.add_argument('--attention', action='store_true',
                       help='Run attention convergence experiments')
    parser.add_argument('--rotation', action='store_true',
                       help='Run rotation equivariance experiments')
    parser.add_argument('--seed', type=int, default=None,
                       help=f'Random seed (default: {SEED})')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')
    
    return parser.parse_args()


def main():
    # main entry point
    args = parse_args()
    
    # Override seed if specified
    if args.seed is not None:
        import src.config as config
        config.SEED = args.seed
    
    # Set seed for reproducibility
    set_seed(SEED if args.seed is None else args.seed)
    
    verbose = not args.quiet
    
    # Run specific experiments or all
    specific = args.mnist or args.pde or args.attention or args.rotation
    
    if not specific:
        run_all(verbose)
    else:
        print_banner()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        if args.mnist:
            run_mnist(verbose)
        if args.pde:
            run_pde(verbose)
        if args.attention:
            run_attention(verbose)
        if args.rotation:
            run_rotation(verbose)
        
        print("\n" + "=" * 70)
        print(f"Results saved to: {RESULTS_DIR}")
        print("=" * 70)


if __name__ == "__main__":
    main()
