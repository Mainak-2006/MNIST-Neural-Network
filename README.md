# MNIST from scratch

A handwritten-digit classifier built from the ground up with **only NumPy**.
No PyTorch, no TensorFlow, no autograd. Every forward pass, gradient, and
weight update is written out so you can read the mathematics directly in
the code.

## The task

Given a 28x28 grayscale image of a handwritten digit, predict which digit
(0-9) it is. MNIST has 60,000 training and 10,000 test images.

## The model

A multi-layer perceptron (MLP):

```
784 inputs -> Linear -> ReLU -> Linear -> ReLU -> 10 outputs -> softmax + cross-entropy
```

- **Linear** produces `Z = XW + b`.
- **ReLU** adds non-linearity: `max(0, Z)`. Without it, stacking linear
  layers would collapse into a single linear layer.
- **Softmax** turns 10 raw scores (logits) into a probability distribution.
- **Cross-entropy loss** measures how far those probabilities are from the
  one-hot truth.

## The math

For a Linear layer with weight `W` and bias `b` on a batch `X`:

| Forward | Backward (given `dL/dZ` from above) |
|---|---|
| `Z = XW + b` | `dL/dW = Xᵀ (dL/dZ)` |
| | `dL/db = Σ_batch (dL/dZ)` |
| | `dL/dX = (dL/dZ) Wᵀ` |

ReLU gates the gradient with its own derivative:
`dL/dZ = (dL/dA) · (Z > 0)`.

The softmax + cross-entropy pair has a famously clean gradient:
`dL/dZ = (p - y) / B`, where `p` is the predicted probability and `y` the
one-hot label. It says: raise the probability of the true class, lower the
rest, in proportion to the current error.

Weights use **He initialisation** (std `sqrt(2 / fan_in)`) so activation
variance stays roughly constant across layers. Training uses **mini-batch
SGD with momentum**, and **Adam** is also available.

`grad_check.py` verifies the backprop above against finite-difference
numerical gradients. This is the single best sanity check for a hand-written
network, and it should pass before you trust any training run.

## Files

| File | Purpose |
|---|---|
| `data.py` | Download + load MNIST, normalize, flatten, one-hot encode |
| `network.py` | `Linear`, `ReLU`, `SoftmaxCrossEntropy`, `MLP`, optimisers |
| `grad_check.py` | Verify backprop against numerical gradients |
| `train.py` | Mini-batch training loop, evaluation, checkpointing |
| `predict.py` | Test N digits at once, or your own image file(s) |
| `visualize.py` | Sample digits, training curves, misclassified examples |
| `animate_training.py` | Record training and export `training.gif` + `training.mp4` |
| `animate_predict.py` | Animate batch inference and export `predict.gif` + `predict.mp4` |

## Usage

```bash
pip install -r requirements.txt

python grad_check.py          # confirm the gradients are correct
python train.py               # train (defaults: SGD, 256-128, 15 epochs)
python visualize.py           # produce artifacts/plots.png
python animate_training.py    # produce artifacts/training.gif + training.mp4
python predict.py             # test 100 random digits at once, save a grid
python predict.py --image my_digit.png --center   # test your own image
python animate_predict.py     # produce artifacts/predict.gif + predict.mp4
```

Options:

```bash
python train.py --optimizer adam --lr 1e-3 --epochs 10
python train.py --hidden 512 256 --lr 0.05 --batch-size 256
python train.py --weight-decay 1e-4 --lr-decay 0.95
```

Expect roughly **98%** test accuracy after ~15 epochs.

## Ideas to extend

- Add a `Dropout` layer between the hidden layers.
- Swap SGD for a learning-rate schedule or add Nesterov momentum.
- Add a second hidden layer and study the bias/variance trade-off.
- Replace the MLP with a small convolutional network (a much bigger step).
- Visualise the learned first-layer weights to see the learned features.
