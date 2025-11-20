import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
from scipy.sparse.linalg import cg
from scipy.sparse import diags
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SIREN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.layers[0].weight.uniform_(-1/2, 1/2)
            for layer in self.layers[1:]:
                bound = np.sqrt(6 / layer.weight.shape[1]) / 30
                layer.weight.uniform_(-bound, bound)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.sin(30.0 * layer(x)) if i == 0 else torch.sin(layer(x))
        return self.layers[-1](x)

    def set_parameters(self, params):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data = params[idx:idx+n].reshape(p.shape)
            idx += n


class TransformerOperator(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, target_model_params=None):
        super().__init__()
        self.input_embed = nn.Linear(3, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if target_model_params is None:
            target_model_params = 2*64 + 64 + 64*64 + 64 + 64*64 + 64 + 64*1 + 1
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, target_model_params)
        )

    def forward(self, S):
        coords, values = S[:, :, :2], S[:, :, 2:3]
        mu = coords.mean(dim=1, keepdim=True)
        centered = coords - mu
        s = torch.sqrt((centered**2).mean(dim=[1,2], keepdim=True)).clamp(min=1e-6)
        coords_norm = centered / s
        tokens = torch.cat([coords_norm, values], dim=-1)
        embedded = self.input_embed(tokens)
        out = self.transformer(embedded)
        pooled = out.mean(dim=1)
        return self.output_head(pooled)


def solve_poisson_2d(n=28, n_sources=3):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X, Y], axis=-1).reshape(-1,2)

    f = np.zeros((n,n))
    for _ in range(n_sources):
        cx, cy = np.random.rand(2)
        sx, sy = 0.05+0.1*np.random.rand(2)
        f += np.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2)))
    f = f.reshape(-1)

    N = n*n
    main = 4*np.ones(N)
    off1 = -1*np.ones(N-1)
    off1[np.arange(1,n)*n-1] = 0
    offn = -1*np.ones(N-n)
    A = diags([main, off1, off1, offn, offn],[0,-1,1,-n,n], format="csr")

    u, info = cg(A, f)
    if info != 0:
        raise RuntimeError("CG did not converge")
    u = u.reshape(n,n)

    samples = np.concatenate([coords, u.reshape(-1,1)], axis=1)
    return torch.tensor(samples, dtype=torch.float32).to(device)


def load_pde_field():
    return solve_poisson_2d(n=28, n_sources=3)


def train_operator(operator, num_iterations=200, lr=3e-4):
    opt = optim.AdamW(operator.parameters(), lr=lr, weight_decay=5e-5)
    losses = []
    for it in range(num_iterations):
        samples = load_pde_field()
        S = samples.unsqueeze(0)
        pred_params = operator(S)
        pred_siren = SIREN().to(device)
        pred_siren.set_parameters(pred_params[0])

        with torch.no_grad():
            y_true = S[0,:,2:3]
        y_pred = pred_siren(S[0,:,:2])
        loss = nn.functional.mse_loss(y_pred, y_true)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def validate_affine_equivariance(operator, a=1.5, b=(0.3,-0.3), return_fields=False):
    samples = load_pde_field()
    coords, values = samples[:,:2], samples[:,2:3]
    b = torch.tensor(b, device=device).float()

    coords_t = coords*a + b
    samples_t = torch.cat([coords_t, values], dim=-1)

    with torch.no_grad():
        params1 = operator(samples.unsqueeze(0))
        f_S = SIREN().to(device)
        f_S.set_parameters(params1[0])

        params2 = operator(samples_t.unsqueeze(0))
        f_Sp = SIREN().to(device)
        f_Sp.set_parameters(params2[0])

    grid = torch.tensor(
        np.stack(np.meshgrid(np.linspace(-2,2,64), np.linspace(-2,2,64)),-1).reshape(-1,2),
        dtype=torch.float32, device=device
    )

    with torch.no_grad():
        field_A = f_Sp(grid).reshape(64,64).cpu().numpy()
        field_B = f_S(((grid-b)/a)).reshape(64,64).cpu().numpy()

    mse = np.mean((field_A - field_B)**2)

    if return_fields:
        return mse, field_A, field_B
    else:
        return mse


def run_trials(n_trials=5):
    siren_params = sum(p.numel() for p in SIREN().parameters())
    mses = []
    field_As, field_Bs = [], []

    for t in range(n_trials):
        print(f"\n=== Trial {t+1}/{n_trials} ===")
        torch.manual_seed(SEED+t)
        np.random.seed(SEED+t)
        operator = TransformerOperator(target_model_params=siren_params).to(device)
        train_operator(operator, num_iterations=200)

        mse, field_A, field_B = validate_affine_equivariance(operator, return_fields=True)
        mses.append(mse)
        field_As.append(field_A)
        field_Bs.append(field_B)

    mean, sem = np.mean(mses), stats.sem(mses)
    ci95 = sem*stats.t.ppf(0.975, n_trials-1)
    print(f"\nFinal MSE over {n_trials} trials: {mean:.6f} ± {ci95:.6f} (95% CI)")

    field_A_mean = np.mean(np.stack(field_As, axis=0), axis=0)
    field_B_mean = np.mean(np.stack(field_Bs, axis=0), axis=0)
    diff = np.abs(field_A_mean - field_B_mean)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(field_A_mean); plt.title("Avg f_S'(x)")
    plt.subplot(1,3,2); plt.imshow(field_B_mean); plt.title("Avg f_S((x-b)/a)")
    plt.subplot(1,3,3); plt.imshow(diff); plt.title(f"Avg Diff (MSE={mean:.4f}±{ci95:.4f})")

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/2DPDEFigure.png", dpi=300)
    plt.show()

    return mses


if __name__ == "__main__":
    run_trials(n_trials=10)
