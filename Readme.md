# Neural Fields Meet Attention

This project examines how attention mechanisms relate to neural field optimization. The paper shows that a transformer with relative positional information and coordinate normalization preserves affine structure when it is used as a set to function operator. It also shows that linear attention produces exact negative gradients for sinusoidal fields and that softmax attention converges to the same result at high temperature.

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
# Neural Fields Meet Attention

This repository contains code and figure generation scripts supporting the paper "Neural Fields Meet Attention" (NeurIPS 2025 workshop submission). The code reproduces experiments demonstrating affine equivariance properties of attention-based operators and visualizes rotation equivariance on SO(2) (2D) and SO(3) (sphere) fields.

This README documents how to set up, run, and reproduce figures and experiments in this repository.

## Requirements and Installation

1. Create and activate a Python virtual environment. Example (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Or (cmd):

```cmd
venv\Scripts\activate
```

On macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies listed in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Notes:
- If you require GPU support for PyTorch, install the correct CUDA-enabled wheel for your system. See https://pytorch.org/get-started/locally/ for the platform-correct command (e.g., `pip` wheel with `+cu129` tag for CUDA 12.9). The rest of the packages can be installed via `pip install -r requirements.txt`.

## Project Layout

Top-level layout (key files):

```
nfma-cr/
├── src/
│   ├── config.py                # Centralized configuration and defaults
│   ├── experiments/
│   │   └── rotation_equivariance.py  # SO(2) and SO(3) figure generation
+│   ├── models/                  # SIREN & Transformer model code
│   └── utils/                   # plotting, saving, seed, data utilities
├── results/                     # Auto-created; generated figures saved here
├── requirements.txt             # Python dependencies (pinned)
├── Readme.md                    # This file
└── main.py                      # CLI entry point to run experiments
```

## Quick Start

- Generate the rotation equivariance figures (SO(2) and SO(3)):

```powershell
python main.py --rotation
```

- Run all experiments (MNIST, PDE, attention convergence, rotation figures):

```powershell
python main.py --all
```

## Figure generation and customization

- Figure and visualization parameters are controlled from `src/config.py`. Examples of commonly edited keys:
  - `FIGURE_DPI`, `FIGURE_FORMAT` — output image settings
  - `SO3_MESH_U`, `SO3_MESH_V` — sphere mesh resolution
  - `SO3_WIREFRAME_RSTRIDE`, `SO3_WIREFRAME_CSTRIDE` — wireframe density
  - `SO3_BACKDROP_OFFSET`, `SO3_BACKDROP_ALPHA` — grid backdrop position and visibility

To customize the camera angle, axis labels, or other styling, edit `src/experiments/rotation_equivariance.py` or add a small wrapper that calls the relevant plotting function.

## Reproducing Figures for Publication

1. Ensure `src/config.py` contains the desired figure settings (DPI, format, mesh resolution).
2. Run the specific experiments or the figure-generation target:

```powershell
python main.py --rotation
```

3. Output images are saved to the `results/` directory. Example filenames:
  - `so2_rotation_equivariance.png`
  - `so3_rotation_equivariance.png`

## Troubleshooting

- If matplotlib raises rendering errors when using high-resolution spheres, reduce `SO3_MESH_U` / `SO3_MESH_V` in `src/config.py`.
- For GPU acceleration with PyTorch: install the CUDA-matched PyTorch wheel from pytorch.org. Using an incompatible wheel may cause import errors.
- If figures are clipped when saving, try increasing `pad_inches` or using `bbox_inches='tight'` in `src/utils/visualization.py::save_figure`.

## Poster Link

If you would like to see the poster accompanying this paper at NeurIPS 2025, please see it [here]([url](https://docs.google.com/presentation/d/1ggmoxG2ShRdEyLaTONrtkOY8swQjNQbi/edit?slide=id.p1#slide=id.p1))!

## Demo Video

As a part of the Symmetry and Geometry in Neural Representations workshop, we have also contributed a video demo of this research to be featured publicly on their platform. That video can also be accessed [here](url).

## Citation

If you use this code or figures in your work, please cite the corresponding paper:

```bibtex
@inproceedings{cherukuri2025neural,
  title     = {Neural Fields Meet Attention},
  author    = {Cherukuri, Kalyan and Lala, Aarav},
  booktitle = {NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations},
  year      = {2025}
}
```

## Contributing

Contributions and issue reports are welcome. When proposing a change, please include a short description of the intent, the plot or figure it affects, and a screenshot of the current/desired output if relevant.
