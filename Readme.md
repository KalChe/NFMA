# Neural Fields Meet Attention

This project examines how attention mechanisms relate to neural field optimization. The paper shows that a transformer with relative positional information and coordinate normalization preserves affine structure when it is used as a set to function operator. It also shows that linear attention produces exact negative gradients for sinusoidal fields and that softmax attention converges to the same result at high temperature.

## Setup

Run the following commands in a terminal:

python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt

## File structure

Appendix F  
Extended_Validation\MNIST_as_field.py

Appendix G  
Extended_Validation\PDEGrid.py

