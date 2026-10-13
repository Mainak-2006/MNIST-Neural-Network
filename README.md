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

`Dropout` (inverted, train-only) can be inserted between the hidden layers to
regularise the model. A small convolutional network is also included — see
`convnet.py` and `--model cnn` below.

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
SGD with momentum** (Nesterov optional), and **Adam** is also available. A
learning-rate schedule (`exp`, `step`, or `cosine`) can be applied per epoch.

`grad_check.py` verifies the backprop above against finite-difference
numerical gradients — for both the MLP and the convolutional network. This is
the single best sanity check for a hand-written network, and it should pass
before you trust any training run.

## The convolutional network

`convnet.py` keeps the 2-D image layout and slides small learnable filters
over it:

```
1x28x28 -> Conv(8,3x3) -> ReLU -> MaxPool(2) -> Conv(16,3x3) -> ReLU -> MaxPool(2)
        -> Flatten -> Linear(128) -> ReLU -> Linear(10) -> softmax + cross-entropy
```

`Conv2d` is implemented with *im2col* so the forward pass is a single matrix
multiply, and `MaxPool2d` routes gradients back to each window's argmax. Both
are checked with finite differences in `grad_check.py`.

## Files

| File | Purpose |
|---|---|
| `data.py` | Download + load MNIST, normalize, flatten, one-hot encode |
| `network.py` | `Linear`, `ReLU`, `Dropout`, `SoftmaxCrossEntropy`, `MLP`, optimisers |
| `convnet.py` | `Conv2d`, `MaxPool2d`, `Flatten`, `ConvNet` (a small CNN) |
| `models.py` | `load_model` — dispatch a checkpoint to the MLP or CNN loader |
| `grad_check.py` | Verify backprop against numerical gradients (MLP + CNN) |
| `train.py` | Mini-batch training loop (MLP or CNN), evaluation, checkpointing |
| `predict.py` | Test N digits at once, or your own image file(s) |
| `visualize.py` | Sample digits, training curves, mistakes, first-layer features |
| `experiments.py` | Depth / capacity study (bias vs variance) |
| `animate_training.py` | Record training and export `training.gif` + `training.mp4` |
| `animate_predict.py` | Animate batch inference and export `predict.gif` + `predict.mp4` |

## Usage

```bash
pip install -r requirements.txt

python grad_check.py          # confirm the gradients are correct (MLP + CNN)
python train.py               # train (defaults: MLP, SGD, 256-128, 15 epochs)
python visualize.py           # produce artifacts/plots.png + weights.png
python animate_training.py    # produce artifacts/training.gif + training.mp4
python predict.py             # test 100 random digits at once, save a grid
python predict.py --image my_digit.png --center   # test your own image
python animate_predict.py     # produce artifacts/predict.gif + predict.mp4
```

Options:

```bash
python train.py --optimizer adam --lr 1e-3 --epochs 10
python train.py --hidden 512 256 --lr 0.05 --batch-size 256
python train.py --dropout 0.2 --nesterov --lr-schedule cosine
python train.py --weight-decay 1e-4 --lr-decay 0.95
python train.py --model cnn --optimizer adam --epochs 8   # convolutional net
python experiments.py         # 1/2/3 hidden layers, bias-variance
```

Training a model also drives `predict.py` and `visualize.py` for either model
type — `models.load_model` reads the checkpoint's `kind` and picks the right
network. The depth study writes `artifacts/depth_study.png`.

Expect roughly **98%** test accuracy after ~15 epochs for the MLP, and the CNN
reaches similar accuracy with fewer epochs (at higher per-epoch cost).

## Ideas to extend

- Add `Dropout` to the CNN and compare it against the MLP.
- Train the deeper/convolutional models to convergence and study the curves.
- Add data augmentation (small random shifts) to make the CNN more robust.
- Replace max pooling with average pooling or add a second conv block.
