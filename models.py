"""Load a saved model without knowing whether it is an MLP or a ConvNet.

Both models write a `kind` tag into their .npz checkpoint ("mlp" or "cnn").
This helper reads that tag and dispatches to the right loader, so predict.py
and visualize.py work with either model transparently.
"""

import numpy as np


def load_model(path):
    """Return the MLP or ConvNet saved at `path`."""
    with np.load(path) as data:
        kind = str(data["kind"]) if "kind" in data.files else "mlp"
    if kind == "cnn":
        from convnet import ConvNet
        return ConvNet.load(path)
    from network import MLP
    return MLP.load(path)


def model_supports_images(model):
    """True if the model takes (B, 1, 28, 28) feature maps rather than flat rows."""
    return model.__class__.__name__ == "ConvNet"


def as_model_input(model, x_flat):
    """Reshape flat (B, 784) rows into whatever the model expects."""
    if model_supports_images(model):
        return x_flat.reshape(len(x_flat), 1, 28, 28)
    return x_flat
