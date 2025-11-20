# Neural Fields Meet Attention

This project examines how attention mechanisms relate to neural field optimization. The paper shows that a transformer with relative positional information and coordinate normalization preserves affine structure when it is used as a set to function operator. It also shows that linear attention produces exact negative gradients for sinusoidal fields and that softmax attention converges to the same result at high temperature.

## Setup

Run the following commands in a terminal:
1. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:

    ```bash
    venv\Scripts\activate
    ```
3. **Install dependencies**:
    
    ```bash
    pip install numpy
    ```
## File structure

## Appendix F. Extended Validation on a Computer Vision Task  
  
    ```bash
    Extended_Validation\MNIST_as_field.py
    ```

##Appendix G. Extended Validation on a Physics Task  

    ```bash
    Extended_Validation\PDEGrid.py
        ```

## License and Citation

This code is part of the Neural Fields Meet Attention research project. If you use this code or the generated figures, please cite the original paper.

@inproceedings{cherukuri2025neural,
  title={Neural Fields Meet Attention},
  author={Cherukuri, Kalyan and Lala, Aarav},
  booktitle={NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations}
}