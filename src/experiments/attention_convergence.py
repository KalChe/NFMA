# attention convergence experiments
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from ..config import DEVICE, SEED, RESULTS_DIR
from ..utils import set_seed, save_figure


# publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
})


def softmax_attention(Q, K, V, temperature=1.0):
    # softmax attention with temperature scaling
    d = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d)
    weights = F.softmax(scores / temperature, dim=-1)
    return weights @ V


def uniform_attention(V):
    # uniform attention (τ → ∞ limit)
    N = V.shape[0]
    return V.mean(dim=0, keepdim=True).expand(N, -1)


def run_convergence_experiment(n_trials: int = 10):
    # test softmax -> uniform convergence at o(tau^-2);
    # returns dictionary with mse and uniformity results per temperature
    temperatures = [1, 10, 100, 1000]
    results = {t: {'mse': [], 'uniformity': []} for t in temperatures}
    
    N, d = 64, 32
    
    for trial in range(n_trials):
        set_seed(SEED + trial)
        
        Q = torch.randn(N, d, device=DEVICE)
        K = torch.randn(N, d, device=DEVICE)
        V = torch.randn(N, 1, device=DEVICE)
        
        uniform_out = uniform_attention(V)
        
        for temp in temperatures:
            softmax_out = softmax_attention(Q, K, V, temperature=temp)
            
            # MSE to uniform
            mse = F.mse_loss(softmax_out, uniform_out).item()
            results[temp]['mse'].append(mse)
            
            # Uniformity: how constant is output
            s_flat = softmax_out.squeeze()
            cv = (s_flat.std() / (s_flat.mean().abs() + 1e-10)).item()
            uniformity = 1.0 / (1.0 + cv)
            results[temp]['uniformity'].append(uniformity)
    
    # Aggregate
    summary = {}
    for temp in temperatures:
        summary[temp] = {
            'mse_mean': np.mean(results[temp]['mse']),
            'mse_std': np.std(results[temp]['mse']),
            'uniformity_mean': np.mean(results[temp]['uniformity']),
            'uniformity_std': np.std(results[temp]['uniformity'])
        }
    
    return summary, temperatures


def run_attention_convergence_experiments(n_trials: int = 10, verbose: bool = True):
    # run all attention convergence experiments and create figures
    if verbose:
        print("\n  Running softmax convergence experiment...")
    
    summary, temps = run_convergence_experiment(n_trials)
    
    # Create main figure: 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Detailed temperature sweep
    set_seed(SEED)
    temp_range = np.logspace(0, 3, 50)
    mse_values = []
    
    N, d = 64, 32
    Q = torch.randn(N, d, device=DEVICE)
    K = torch.randn(N, d, device=DEVICE)
    V = torch.randn(N, 1, device=DEVICE)
    uniform_out = uniform_attention(V)
    
    for temp in temp_range:
        softmax_out = softmax_attention(Q, K, V, temperature=temp)
        mse = F.mse_loss(softmax_out, uniform_out).item()
        mse_values.append(mse)
    
    mse_values = np.array(mse_values)
    mse_norm = mse_values / mse_values[0]
    
    # Theory
    theory_tau2 = 1.0 / (temp_range ** 2)
    theory_tau2_norm = theory_tau2 / theory_tau2[0]
    
    # Panel (a): MSE decay
    ax = axes[0]
    ax.loglog(temp_range, mse_norm, 'b-', linewidth=2.5, label='Empirical', 
              marker='o', markersize=4, markevery=5)
    ax.loglog(temp_range, theory_tau2_norm, 'r--', linewidth=2, label=r'Theory $O(\tau^{-2})$')
    ax.set_xlabel(r'Temperature $\tau$', fontsize=12)
    ax.set_ylabel('Normalized MSE to Uniform', fontsize=12)
    ax.set_title(r'MSE Decays as $O(\tau^{-2})$', fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1, 1000)
    
    # Panel (b): Uniformity
    ax = axes[1]
    u_means = [summary[t]['uniformity_mean'] for t in temps]
    u_stds = [summary[t]['uniformity_std'] for t in temps]
    
    ax.errorbar(temps, u_means, yerr=u_stds, fmt='bo-', linewidth=2,
                markersize=10, capsize=5, capthick=2, label='Empirical')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='Uniform Limit')
    ax.set_xscale('log')
    ax.set_xlabel(r'Temperature $\tau$', fontsize=12)
    ax.set_ylabel('Uniformity Score', fontsize=12)
    ax.set_title(r'Output Approaches Uniform as $\tau \to \infty$', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 1500)
    ax.set_ylim(0.0, 1.05)
    
    fig.suptitle(r'Softmax Converges to Uniform Attention at $O(\tau^{-2})$', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'softmax_convergence')
    plt.close(fig)
    
    if verbose:
        print("\n  Convergence Results:")
        print("  " + "-" * 60)
        prev_mse = None
        for t in temps:
            s = summary[t]
            scaling = ""
            if prev_mse is not None and s['mse_mean'] > 0:
                ratio = prev_mse / s['mse_mean']
                exponent = np.log10(ratio) / np.log10(10)
                scaling = f"~τ^{-exponent:.2f}"
            print(f"  τ={t:<4}: uniformity={s['uniformity_mean']:.4f}, "
                  f"MSE={s['mse_mean']:.2e} {scaling}")
            prev_mse = s['mse_mean']
        print("  " + "-" * 60)
    
    return summary


def create_temperature_scaling_figure(verbose: bool = True):
    # create 2d vs 3d temperature scaling comparison figure
    if verbose:
        print("\n  Creating temperature scaling comparison...")
    
    set_seed(SEED)
    temp_range = np.logspace(0, 3, 50)
    
    # Theory reference
    theory_tau2 = 1.0 / (temp_range ** 2)
    theory_tau2_norm = theory_tau2 / theory_tau2[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel (a): 2D vs 3D comparison
    ax = axes[0]
    
    # 2D field
    N_2d, d_2d = 64, 32
    Q_2d = torch.randn(N_2d, d_2d, device=DEVICE)
    K_2d = torch.randn(N_2d, d_2d, device=DEVICE)
    V_2d = torch.randn(N_2d, 1, device=DEVICE)
    uniform_2d = uniform_attention(V_2d)
    
    mse_2d = []
    for temp in temp_range:
        soft_2d = softmax_attention(Q_2d, K_2d, V_2d, temperature=temp)
        mse_2d.append(F.mse_loss(soft_2d, uniform_2d).item())
    
    # 3D field (larger dimension)
    N_3d, d_3d = 64, 48
    Q_3d = torch.randn(N_3d, d_3d, device=DEVICE)
    K_3d = torch.randn(N_3d, d_3d, device=DEVICE)
    V_3d = torch.randn(N_3d, 1, device=DEVICE)
    uniform_3d = uniform_attention(V_3d)
    
    mse_3d = []
    for temp in temp_range:
        soft_3d = softmax_attention(Q_3d, K_3d, V_3d, temperature=temp)
        mse_3d.append(F.mse_loss(soft_3d, uniform_3d).item())
    
    mse_2d = np.array(mse_2d)
    mse_3d = np.array(mse_3d)
    mse_2d_norm = mse_2d / mse_2d[0]
    mse_3d_norm = mse_3d / mse_3d[0]
    
    ax.loglog(temp_range, mse_2d_norm, 'b-', linewidth=2.5, label='2D Field (d=32)',
              marker='s', markersize=4, markevery=5)
    ax.loglog(temp_range, mse_3d_norm, 'g-', linewidth=2.5, label='3D Field (d=48)',
              marker='^', markersize=4, markevery=5)
    ax.loglog(temp_range, theory_tau2_norm, 'r--', linewidth=2, label=r'Theory $O(\tau^{-2})$')
    
    ax.set_xlabel(r'Temperature $\tau$', fontsize=12)
    ax.set_ylabel('Normalized MSE to Uniform', fontsize=12)
    ax.set_title('Dimension-Independent Convergence Rate', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1, 1000)
    
    # Panel (b): Convergence verification
    ax = axes[1]
    
    N, d = 64, 32
    Q = torch.randn(N, d, device=DEVICE)
    K = torch.randn(N, d, device=DEVICE)
    V = torch.randn(N, 1, device=DEVICE)
    uniform_out = uniform_attention(V)
    
    mse_values = []
    for temp in temp_range:
        softmax_out = softmax_attention(Q, K, V, temperature=temp)
        mse = F.mse_loss(softmax_out, uniform_out).item()
        mse_values.append(mse)
    
    mse_values = np.array(mse_values)
    mse_norm = mse_values / mse_values[0]
    
    ax.loglog(temp_range, mse_norm, 'b-', linewidth=2.5, label='Empirical',
              marker='o', markersize=4, markevery=5)
    ax.loglog(temp_range, theory_tau2_norm, 'r--', linewidth=2, label=r'Theory $O(\tau^{-2})$')
    
    ax.set_xlabel(r'Temperature $\tau$', fontsize=12)
    ax.set_ylabel('Normalized MSE to Uniform', fontsize=12)
    ax.set_title(r'Softmax $\to$ Uniform: $O(\tau^{-2})$ Convergence', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1, 1000)
    
    fig.suptitle('Temperature Scaling Validation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'temperature_scaling')
    plt.close(fig)
    
    return fig
