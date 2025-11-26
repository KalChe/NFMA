# configuration settings for neural fields experiments
# hyperparameters and paths are centralized here for reproducibility
import os
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SIREN Network
SIREN_HIDDEN_DIM = 64
SIREN_NUM_LAYERS = 3
SIREN_OMEGA_0 = 30.0

# Transformer Operator
TRANSFORMER_EMBED_DIM = 128
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-5
NUM_ITERATIONS = 200

# Number of trials for statistical significance
NUM_TRIALS = 5

# Grid settings for visualization
GRID_SIZE = 64
GRID_RANGE = (-2, 2)

# MNIST settings
MNIST_SIZE = 28

# PDE settings
PDE_GRID_SIZE = 28
PDE_NUM_SOURCES = 3

# Affine transformation defaults
AFFINE_SCALE = 1.5
AFFINE_TRANSLATION = (0.3, -0.3)

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

SO3_PLANE_RES = 40
SO3_PLANE_SIZE = 1.2
SO3_N_BINS = 10
SO3_WIREFRAME = True
SO3_WIREFRAME_RSTRIDE = 3
SO3_WIREFRAME_CSTRIDE = 3

SO3_MESH_U = 160
SO3_MESH_V = 80
SO3_ENABLE_LIGHTING = True
SO3_USE_TRISURF = False
SO3_AXIS_LINEWIDTH = 1.2

SO3_BACKDROP_ENABLE = True
SO3_BACKDROP_OFFSET = -1.45
SO3_BACKDROP_COLOR = '#888888'
SO3_BACKDROP_ALPHA = 0.45
SO3_BACKDROP_RSTRIDE = 6
SO3_BACKDROP_CSTRIDE = 6
SO3_BACKDROP_TICKS = 5
