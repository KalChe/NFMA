# data utilities
import torch
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags

from ..config import (
    DEVICE, SEED, DATA_DIR, GRID_SIZE, GRID_RANGE, 
    MNIST_SIZE, PDE_GRID_SIZE, PDE_NUM_SOURCES
)


def set_seed(seed: int = SEED):
    # set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_grid(
    size: int = GRID_SIZE, 
    range_min: float = GRID_RANGE[0], 
    range_max: float = GRID_RANGE[1]
) -> torch.Tensor:
    # create a 2d grid of coordinates
    x = np.linspace(range_min, range_max, size)
    y = np.linspace(range_min, range_max, size)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    return torch.tensor(grid, dtype=torch.float32, device=DEVICE)


def load_mnist_field(batch_size: int = 1):
    # load mnist digit as a continuous field; returns samples (x,y,intensity) & label
    import torchvision
    import torchvision.transforms as T
    
    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root=str(DATA_DIR), 
        train=True, 
        download=True, 
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    imgs, labels = next(iter(loader))
    img = imgs[0, 0].numpy()
    
    # coordinate grid [-1, 1]^2
    coords = np.stack(
        np.meshgrid(
            np.linspace(-1, 1, MNIST_SIZE), 
            np.linspace(-1, 1, MNIST_SIZE)
        ), 
        axis=-1
    ).reshape(-1, 2)
    
    values = img.reshape(-1, 1)
    samples = np.concatenate([coords, values], axis=1)
    
    return torch.tensor(samples, dtype=torch.float32, device=DEVICE), labels[0].item()


def solve_poisson_2d(n: int = PDE_GRID_SIZE, n_sources: int = PDE_NUM_SOURCES):
    # solve 2d poisson equation with random gaussian sources; returns samples (x,y,solution)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)
    
    # random gaussian sources
    f = np.zeros((n, n))
    for _ in range(n_sources):
        cx, cy = np.random.rand(2)
        sx, sy = 0.05 + 0.1 * np.random.rand(2)
        f += np.exp(-((X - cx)**2 / (2 * sx**2) + (Y - cy)**2 / (2 * sy**2)))
    f = f.reshape(-1)
    
    # build laplacian operator
    N = n * n
    main = 4 * np.ones(N)
    off1 = -1 * np.ones(N - 1)
    off1[np.arange(1, n) * n - 1] = 0
    offn = -1 * np.ones(N - n)
    A = diags([main, off1, off1, offn, offn], [0, -1, 1, -n, n], format="csr")
    
    # Solve with conjugate gradient
    u, info = cg(A, f)
    if info != 0:
        raise RuntimeError("CG solver did not converge")
    u = u.reshape(n, n)
    
    samples = np.concatenate([coords, u.reshape(-1, 1)], axis=1)
    return torch.tensor(samples, dtype=torch.float32, device=DEVICE)
