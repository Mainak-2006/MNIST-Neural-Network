"""Load the MNIST dataset.

MNIST is a set of 70,000 grayscale images of handwritten digits (0-9):
  - 60,000 for training, 10,000 for testing
  - each image is 28x28 pixels, each pixel an integer in [0, 255]

We store it as a single .npz archive with four arrays:
  x_train (60000, 28, 28), y_train (60000,)
  x_test  (10000, 28, 28), y_test  (10000,)

Two preprocessing steps matter a lot for training:

  1. Flatten 28x28 -> 784. Our network's first layer is a matrix-vector
     product, so it expects each image as one long vector of 784 numbers.

  2. Scale pixels from [0, 255] to [0, 1]. Gradient descent moves in
     proportion to the input magnitude. If inputs are huge and uneven,
     the loss surface becomes a stretched, skinny valley and training
     crawls or diverges. Putting every input on a small, comparable
     range makes optimisation dramatically easier.

  3. One-hot encode the labels. The target for a digit is a 10-vector
     with a 1 in the correct position and 0 elsewhere, e.g. 3 -> 
     [0,0,0,1,0,0,0,0,0,0]. This pairs with the softmax cross-entropy
     loss we use in network.py.
"""

import os
import urllib.request

import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PATH = os.path.join(DATA_DIR, "mnist.npz")


def download():
    """Download mnist.npz once and cache it under ./data/."""
    if os.path.exists(PATH):
        return PATH
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading MNIST to {PATH} ...")
    urllib.request.urlretrieve(URL, PATH)
    print("Done.")
    return PATH


def one_hot(y, num_classes=10):
    """Convert integer labels (N,) into one-hot rows (N, num_classes)."""
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def load(flatten=True, normalize=True, one_hot_labels=True):
    """Return ((x_train, y_train), (x_test, y_test)) with the options applied."""
    download()
    with np.load(PATH) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    if flatten:
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)

    if one_hot_labels:
        y_train = one_hot(y_train)
        y_test = one_hot(y_test)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_tr, y_tr), (x_te, y_te) = load()
    print("train x:", x_tr.shape, x_tr.dtype, "y:", y_tr.shape)
    print("test  x:", x_te.shape, x_te.dtype, "y:", y_te.shape)
    print("pixel range:", x_tr.min(), "-", x_tr.max())
